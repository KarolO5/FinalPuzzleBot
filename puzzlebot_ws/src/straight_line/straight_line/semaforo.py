#!/usr/bin/env python3
# =============================================================================
# semaforo.py
# =============================================================================
# Detecta el color del semáforo (rojo, amarillo, verde) usando OpenCV (HSV).
#
# ROI: columna derecha del frame (donde físicamente está el semáforo en pista)
#   - Horizontal : ROI_X_START .. 1.0  (fracción derecha, ej. 60-100 %)
#   - Vertical   : 0 .. ROI_Y_END      (fracción superior, ej. 0-70 %)
#
# Motivo del cambio respecto a la versión anterior:
#   La versión previa usaba la franja superior-central (top 30% × central 50%).
#   Esto generaba falsos positivos con objetos coloridos en el centro de la
#   imagen. El semáforo de la pista se ubica a la derecha del robot, por lo
#   que restringir la ROI al lado derecho elimina ambigüedad y mejora la
#   especificidad de la detección.
#
# EVIDENCIA
#   Guarda hasta MAX_EVIDENCE imágenes por detección en ~/puzzlebot_evidence/semaforo/
#   Mantiene solo las 5 más recientes (rota automáticamente).
#
# TÓPICOS
# ────────
#   Sub : /image/raw          [sensor_msgs/Image]
#   Pub : /semaforo/estado    [std_msgs/String]   ("rojo","amarillo","verde","ninguno")
#   Pub : /semaforo/debug_img [sensor_msgs/Image]
# =============================================================================

import os
import glob
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS DE ROI  (columna derecha del frame)
# ─────────────────────────────────────────────────────────────────────────────
ROI_X_START = 0.60   # inicio horizontal (60 % desde la izquierda)
ROI_Y_END   = 0.70   # fin vertical (top 70 % del frame)

# ─────────────────────────────────────────────────────────────────────────────
# RANGOS HSV DE COLORES
# ─────────────────────────────────────────────────────────────────────────────
RED_LO1  = np.array([  0, 100,  80])
RED_HI1  = np.array([ 10, 255, 255])
RED_LO2  = np.array([165, 100,  80])
RED_HI2  = np.array([180, 255, 255])

YELLOW_LO = np.array([ 18, 100,  80])
YELLOW_HI = np.array([ 35, 255, 255])

GREEN_LO  = np.array([ 40,  80,  60])
GREEN_HI  = np.array([ 90, 255, 255])

MIN_PIXELS = 150

# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCIA
# ─────────────────────────────────────────────────────────────────────────────
MAX_EVIDENCE    = 5
_HOME           = os.environ.get('HOME', '/tmp')
EVIDENCE_DIR    = os.path.join(_HOME, 'puzzlebot_evidence', 'semaforo')

try:
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
except OSError as _e:
    import warnings
    warnings.warn(f'[semaforo] No se pudo crear directorio de evidencia: {_e}')


def _save_evidence(frame: np.ndarray, tag: str) -> None:
    """Guarda una imagen de detección; mantiene solo MAX_EVIDENCE archivos."""
    import time
    try:
        ts   = int(time.time() * 1000)
        path = os.path.join(EVIDENCE_DIR, f'{tag}_{ts}.jpg')
        if not cv2.imwrite(path, frame):
            import sys
            print(f'[semaforo evidencia] cv2.imwrite falló: {path}', file=sys.stderr)
        _rotate_evidence(EVIDENCE_DIR, '*.jpg')
    except Exception as exc:
        import sys
        print(f'[semaforo evidencia] excepción: {exc}', file=sys.stderr)


def _rotate_evidence(directory: str, pattern: str) -> None:
    """Elimina los archivos más antiguos si se supera MAX_EVIDENCE."""
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    while len(files) > MAX_EVIDENCE:
        os.remove(files.pop(0))


# ─────────────────────────────────────────────────────────────────────────────
# NODO
# ─────────────────────────────────────────────────────────────────────────────

class SemaforoNode(Node):

    def __init__(self):
        super().__init__('semaforo')

        qos_be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._bridge       = CvBridge()
        self._prev_estado  = 'ninguno'

        self._pub_estado = self.create_publisher(String, '/semaforo/estado',    10)
        self._pub_debug  = self.create_publisher(Image,  '/semaforo/debug_img', 10)

        self.create_subscription(Image, '/image/raw', self._image_cb, qos_be)

        self.get_logger().info(
            f'SemaforoNode listo | ROI derecha x=[{int(ROI_X_START*100)}%,100%] '
            f'y=[0%,{int(ROI_Y_END*100)}%]'
        )

    def _image_cb(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        estado, debug_frame = self._detect(frame)

        # Guardar evidencia solo cuando aparece una detección nueva (no "ninguno")
        if estado != 'ninguno' and estado != self._prev_estado:
            _save_evidence(debug_frame, estado)
        self._prev_estado = estado

        state_msg      = String()
        state_msg.data = estado
        self._pub_estado.publish(state_msg)

        dbg_msg        = self._bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
        dbg_msg.header = msg.header
        self._pub_debug.publish(dbg_msg)

    def _detect(self, frame: np.ndarray) -> tuple:
        h, w = frame.shape[:2]

        roi_x0 = int(w * ROI_X_START)
        roi_x1 = w
        roi_y0 = 0
        roi_y1 = int(h * ROI_Y_END)

        roi     = frame[roi_y0:roi_y1, roi_x0:roi_x1]
        blurred = cv2.GaussianBlur(roi, (7, 7), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask_red    = (cv2.inRange(hsv, RED_LO1, RED_HI1) |
                       cv2.inRange(hsv, RED_LO2, RED_HI2))
        mask_yellow = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
        mask_green  = cv2.inRange(hsv, GREEN_LO,  GREEN_HI)

        px_red    = int(cv2.countNonZero(mask_red))
        px_yellow = int(cv2.countNonZero(mask_yellow))
        px_green  = int(cv2.countNonZero(mask_green))

        if px_red >= MIN_PIXELS and px_red >= px_yellow and px_red >= px_green:
            estado    = 'rojo'
            box_color = (0, 0, 220)
        elif px_yellow >= MIN_PIXELS and px_yellow >= px_green:
            estado    = 'amarillo'
            box_color = (0, 200, 220)
        elif px_green >= MIN_PIXELS:
            estado    = 'verde'
            box_color = (0, 200, 60)
        else:
            estado    = 'ninguno'
            box_color = (120, 120, 120)

        debug = frame.copy()
        cv2.rectangle(debug, (roi_x0, roi_y0), (roi_x1, roi_y1), box_color, 2)

        label = f'SEMAFORO: {estado.upper()} | R={px_red} A={px_yellow} V={px_green}'
        cv2.putText(debug, label,
                    (roi_x0, min(roi_y1 + 18, h - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        if estado == 'rojo':
            overlay_mask = mask_red
        elif estado == 'amarillo':
            overlay_mask = mask_yellow
        elif estado == 'verde':
            overlay_mask = mask_green
        else:
            overlay_mask = None

        if overlay_mask is not None:
            colored = np.zeros_like(roi)
            colored[overlay_mask > 0] = box_color
            debug[roi_y0:roi_y1, roi_x0:roi_x1] = cv2.addWeighted(
                debug[roi_y0:roi_y1, roi_x0:roi_x1], 0.6, colored, 0.4, 0
            )

        return estado, debug


def main(args=None):
    rclpy.init(args=args)
    node = SemaforoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
