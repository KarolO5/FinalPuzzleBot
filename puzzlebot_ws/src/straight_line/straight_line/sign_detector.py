#!/usr/bin/env python3
# =============================================================================
# sign_detector.py
# =============================================================================
# Nodo ROS 2 para detección de señales de tránsito con modelo YOLO ONNX.
#
# ARQUITECTURA
# ─────────────
# • Suscribe /image/raw, corre inferencia YOLO a 5 Hz.
# • Solo considera detecciones cuyo centro cae en:
#     - Columna izquierda : x ∈ [0,  ROI_LEFT_END]   del frame original
#     - Columna derecha   : x ∈ [ROI_RIGHT_START, 1]  del frame original
#   No se analiza la región central para reducir falsas activaciones.
# • Selecciona la detección de mayor confianza por columna; aplica prioridad
#   global STOP > TurnR > TurnL > AOnly > Give > Crossing.
# • Publica el nombre de la señal más prioritaria, o "ninguna".
#
# CLASES DEL MODELO
# ──────────────────
# CLASS_NAMES define el mapeo índice → etiqueta tal como fue entrenado el
# modelo. Si el orden difiere, editar CLASS_NAMES sin cambiar el resto del nodo.
# Índices esperados: 0=STOP 1=Crossing 2=Give 3=TurnR 4=TurnL 5=AOnly
#
# FORMATO ONNX SOPORTADO
# ───────────────────────
# • YOLOv8 : output shape [1, 4+nc, anchors]   (sin objectness)
# • YOLOv5 : output shape [1, anchors, 5+nc]   (con objectness)
# El parser detecta el formato automáticamente por las dimensiones de salida.
#
# EVIDENCIA
# ──────────
# Guarda hasta 5 imágenes de detección en ~/puzzlebot_evidence/signs/
#
# TÓPICOS
# ────────
#   Sub : /image/raw      [sensor_msgs/Image]
#   Pub : /sign/state     [std_msgs/String]   ("STOP","Crossing","Give",
#                                              "TurnR","TurnL","AOnly","ninguna")
#   Pub : /sign/debug_img [sensor_msgs/Image]
# =============================================================================

import os
import glob
import time
import math

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

# Ruta al modelo.
# Búsqueda en orden:
#   1. Variable de entorno PUZZLEBOT_MODEL_PATH  (mayor prioridad)
#   2. Ruta canónica del workspace del robot en el Docker
#   3. Ruta del repositorio en la Mac de desarrollo
# También puede sobreescribirse en tiempo de ejecución con el parámetro ROS
# 'model_path':  ros2 run straight_line sign_detector --ros-args -p model_path:=/ruta/al/best_copy.onnx
_CANDIDATE_PATHS = [
    os.environ.get('PUZZLEBOT_MODEL_PATH', ''),
    '/home/ubuntu/puzzlebot_docker/puzzlebot_ws/models/best_copy.onnx',
    os.path.expanduser('~/puzzlebot_docker/puzzlebot_ws/models/best_copy.onnx'),
    os.path.expanduser('~/puzzlebot_ws/models/best_copy.onnx'),
]
MODEL_PATH = next(
    (p for p in _CANDIDATE_PATHS if p and os.path.exists(p)),
    '/home/ubuntu/puzzlebot_docker/puzzlebot_ws/models/best_copy.onnx'  # fallback (mostrará error claro)
)

# Clases en el orden en que fueron entrenadas en el modelo
CLASS_NAMES = ['STOP', 'Crossing', 'Give', 'TurnR', 'TurnL', 'AOnly']

# Prioridad (menor índice = mayor prioridad)
SIGN_PRIORITY = ['STOP', 'TurnR', 'TurnL', 'AOnly', 'Give', 'Crossing']

# ROI: columnas izquierda y derecha (fracción del ancho del frame)
ROI_LEFT_END     = 0.20   # columna izquierda: 0 % → 20 %
ROI_RIGHT_START  = 0.80   # columna derecha:  80 % → 100 %

# Inferencia
INFER_INPUT_SIZE = 640    # px (cuadrado)
CONF_THRESH      = 0.45
IOU_THRESH       = 0.45
INFER_HZ         = 5      # Hz máximos de inferencia

# Evidencia
MAX_EVIDENCE  = 5
_HOME         = os.environ.get('HOME', '/tmp')
EVIDENCE_DIR  = os.path.join(_HOME, 'puzzlebot_evidence', 'signs')

try:
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
except OSError as _e:
    import warnings
    warnings.warn(f'[sign_detector] No se pudo crear directorio de evidencia: {_e}')


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE EVIDENCIA
# ─────────────────────────────────────────────────────────────────────────────

def _save_evidence(frame: np.ndarray, tag: str) -> None:
    try:
        ts   = int(time.time() * 1000)
        path = os.path.join(EVIDENCE_DIR, f'{tag}_{ts}.jpg')
        if not cv2.imwrite(path, frame):
            import sys
            print(f'[sign evidencia] cv2.imwrite falló: {path}', file=sys.stderr)
        _rotate(EVIDENCE_DIR, '*.jpg')
    except Exception as exc:
        import sys
        print(f'[sign evidencia] excepción: {exc}', file=sys.stderr)


def _rotate(directory: str, pattern: str) -> None:
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    while len(files) > MAX_EVIDENCE:
        os.remove(files.pop(0))


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCIA YOLO ONNX
# ─────────────────────────────────────────────────────────────────────────────

class YOLODetector:
    """Wrapper ligero para inferencia YOLO v5/v8 con ONNX Runtime."""

    def __init__(self, model_path: str, input_size: int = 640,
                 conf_thresh: float = 0.45, iou_thresh: float = 0.45):
        self.input_size  = input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh

        self.session    = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name

        # Detectar formato (v5 vs v8) por shape del primer output
        out_shape = self.session.get_outputs()[0].shape
        # YOLOv8: [1, 4+nc, anchors] → dim 1 es pequeña
        # YOLOv5: [1, anchors, 5+nc] → dim 1 es grande (25200+)
        self._is_v8 = (len(out_shape) == 3 and out_shape[1] < out_shape[2])

    def detect(self, bgr_frame: np.ndarray) -> list:
        """
        Retorna lista de dicts:
          {'label': str, 'conf': float, 'cx_norm': float, 'cy_norm': float,
           'x1': int, 'y1': int, 'x2': int, 'y2': int}
        Coordenadas en píxeles del frame ORIGINAL.
        """
        orig_h, orig_w = bgr_frame.shape[:2]
        blob, scale, pad = self._preprocess(bgr_frame)

        raw = self.session.run(None, {self.input_name: blob})[0]

        if self._is_v8:
            boxes, scores, class_ids = self._parse_v8(raw)
        else:
            boxes, scores, class_ids = self._parse_v5(raw)

        detections = []
        if len(boxes) == 0:
            return detections

        # NMS
        indices = cv2.dnn.NMSBoxes(
            [b.tolist() for b in boxes],
            scores.tolist(),
            self.conf_thresh,
            self.iou_thresh
        )
        if len(indices) == 0:
            return detections

        for i in (indices.flatten() if hasattr(indices, 'flatten') else indices):
            x1, y1, x2, y2 = self._to_orig(boxes[i], scale, pad, orig_w, orig_h)
            cx_n = ((x1 + x2) / 2) / orig_w
            cy_n = ((y1 + y2) / 2) / orig_h
            cls  = int(class_ids[i])
            label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f'cls{cls}'
            detections.append({
                'label':   label,
                'conf':    float(scores[i]),
                'cx_norm': cx_n,
                'cy_norm': cy_n,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            })

        return detections

    # ── preproceso ────────────────────────────────────────────────────────────

    def _preprocess(self, bgr: np.ndarray):
        """Letterbox resize + normalización. Retorna (blob, scale, pad)."""
        h, w = bgr.shape[:2]
        s    = self.input_size / max(h, w)
        nh, nw = int(h * s), int(w * s)
        resized  = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas   = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_y    = (self.input_size - nh) // 2
        pad_x    = (self.input_size - nw) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        rgb  = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.expand_dims(rgb.transpose(2, 0, 1), 0)
        return blob, s, (pad_x, pad_y)

    # ── parsers ───────────────────────────────────────────────────────────────

    def _parse_v8(self, raw: np.ndarray):
        """YOLOv8: shape [1, 4+nc, anchors]."""
        pred = raw[0].T                          # [anchors, 4+nc]
        nc   = pred.shape[1] - 4
        cls_scores = pred[:, 4:]
        class_ids  = cls_scores.argmax(axis=1)
        scores     = cls_scores.max(axis=1)
        mask       = scores >= self.conf_thresh
        pred, scores, class_ids = pred[mask], scores[mask], class_ids[mask]

        cx, cy = pred[:, 0], pred[:, 1]
        w,  h  = pred[:, 2], pred[:, 3]
        boxes  = np.stack([cx - w/2, cy - h/2, w, h], axis=1)  # xywh
        return boxes, scores, class_ids

    def _parse_v5(self, raw: np.ndarray):
        """YOLOv5: shape [1, anchors, 5+nc]."""
        pred     = raw[0]                        # [anchors, 5+nc]
        obj      = pred[:, 4]
        cls_raw  = pred[:, 5:]
        scores   = obj * cls_raw.max(axis=1)
        class_ids = cls_raw.argmax(axis=1)
        mask      = scores >= self.conf_thresh
        pred, scores, class_ids = pred[mask], scores[mask], class_ids[mask]

        cx, cy = pred[:, 0], pred[:, 1]
        w,  h  = pred[:, 2], pred[:, 3]
        boxes  = np.stack([cx - w/2, cy - h/2, w, h], axis=1)
        return boxes, scores, class_ids

    # ── conversión de coordenadas ─────────────────────────────────────────────

    def _to_orig(self, box, scale, pad, orig_w, orig_h):
        px, py = pad
        x1 = int((box[0] - px) / scale)
        y1 = int((box[1] - py) / scale)
        x2 = int((box[0] + box[2] - px) / scale)
        y2 = int((box[1] + box[3] - py) / scale)
        x1 = max(0, min(x1, orig_w - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        y2 = max(0, min(y2, orig_h - 1))
        return x1, y1, x2, y2


# ─────────────────────────────────────────────────────────────────────────────
# NODO ROS 2
# ─────────────────────────────────────────────────────────────────────────────

class SignDetectorNode(Node):

    def __init__(self):
        super().__init__('sign_detector')

        self._bridge   = CvBridge()
        self._detector = None
        self._last_infer_time = 0.0
        self._infer_period    = 1.0 / INFER_HZ
        self._prev_sign       = 'ninguna'

        qos_be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._pub_state = self.create_publisher(String, '/sign/state',     10)
        self._pub_debug = self.create_publisher(Image,  '/sign/debug_img', 10)

        self.create_subscription(Image, '/image/raw', self._image_cb, qos_be)

        # Parámetro ROS para sobreescribir la ruta del modelo en tiempo de ejecución:
        #   ros2 run straight_line sign_detector --ros-args -p model_path:=/ruta/modelo.onnx
        self.declare_parameter('model_path', MODEL_PATH)
        model_path = self.get_parameter('model_path').get_parameter_value().string_value

        if not _ONNX_AVAILABLE:
            self.get_logger().error(
                'onnxruntime no está instalado. '
                'Instalar con: pip install onnxruntime\n'
                'El nodo publicará "ninguna" hasta que esté disponible.'
            )
            return

        if not os.path.exists(model_path):
            self.get_logger().error(
                f'Modelo no encontrado: {model_path}\n'
                f'  Rutas buscadas automáticamente:\n'
                + '\n'.join(f'    {p}' for p in _CANDIDATE_PATHS if p) +
                f'\n  Solución: ros2 run straight_line sign_detector '
                f'--ros-args -p model_path:=/ruta/absoluta/best_copy.onnx'
            )
            return

        try:
            self._detector = YOLODetector(
                model_path,
                input_size=INFER_INPUT_SIZE,
                conf_thresh=CONF_THRESH,
                iou_thresh=IOU_THRESH,
            )
            self.get_logger().info(
                f'SignDetector listo | modelo={model_path}\n'
                f'  ROI izq=[0-{int(ROI_LEFT_END*100)}%]  '
                f'der=[{int(ROI_RIGHT_START*100)}-100%]\n'
                f'  Inferencia a {INFER_HZ} Hz'
            )
        except Exception as e:
            self.get_logger().error(f'Error cargando modelo: {e}')

    def _image_cb(self, msg: Image):
        now = time.time()
        if now - self._last_infer_time < self._infer_period:
            return
        self._last_infer_time = now

        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        if self._detector is None:
            self._publish_state('ninguna')
            return

        try:
            detections = self._detector.detect(frame)
        except Exception as e:
            self.get_logger().warn(f'Inferencia fallida: {e}')
            self._publish_state('ninguna')
            return

        sign, filtered = self._select_best(detections, frame.shape[1])
        debug_frame    = self._draw_debug(frame, filtered, sign)

        if sign != 'ninguna' and sign != self._prev_sign:
            _save_evidence(debug_frame, sign)
        self._prev_sign = sign

        self._publish_state(sign)

        dbg_msg        = self._bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
        dbg_msg.header = msg.header
        self._pub_debug.publish(dbg_msg)

    # ── selección por ROI y prioridad ─────────────────────────────────────────

    def _select_best(self, detections: list, frame_w: int):
        """Filtra por ROI izq/der y devuelve la señal más prioritaria."""
        in_roi = [
            d for d in detections
            if d['cx_norm'] <= ROI_LEFT_END or d['cx_norm'] >= ROI_RIGHT_START
        ]

        if not in_roi:
            return 'ninguna', []

        # Buscar la señal de mayor prioridad
        for priority_sign in SIGN_PRIORITY:
            candidates = [d for d in in_roi if d['label'] == priority_sign]
            if candidates:
                best = max(candidates, key=lambda d: d['conf'])
                return best['label'], in_roi

        return 'ninguna', in_roi

    # ── visualización ─────────────────────────────────────────────────────────

    def _draw_debug(self, frame: np.ndarray, detections: list, selected: str):
        debug = frame.copy()
        h, w  = frame.shape[:2]

        # Dibujar columnas ROI
        left_x  = int(w * ROI_LEFT_END)
        right_x = int(w * ROI_RIGHT_START)
        cv2.rectangle(debug, (0, 0),       (left_x, h),  (200, 200, 0), 1)
        cv2.rectangle(debug, (right_x, 0), (w, h),        (200, 200, 0), 1)

        colors = {
            'STOP':     (0, 0, 255),
            'Crossing': (255, 165, 0),
            'Give':     (0, 165, 255),
            'TurnR':    (0, 255, 0),
            'TurnL':    (0, 200, 100),
            'AOnly':    (255, 0, 255),
        }

        for d in detections:
            col   = colors.get(d['label'], (200, 200, 200))
            thick = 3 if d['label'] == selected else 1
            cv2.rectangle(debug, (d['x1'], d['y1']), (d['x2'], d['y2']), col, thick)
            cv2.putText(debug,
                        f"{d['label']} {d['conf']:.2f}",
                        (d['x1'], max(d['y1'] - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        txt_col = colors.get(selected, (180, 180, 180))
        cv2.putText(debug, f'SENAL: {selected}',
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_col, 2)

        return debug

    def _publish_state(self, sign: str):
        msg      = String()
        msg.data = sign
        self._pub_state.publish(msg)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = SignDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
