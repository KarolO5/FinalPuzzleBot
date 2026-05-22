#!/usr/bin/env python3
# =============================================================================
# semaforo.py
# =============================================================================
# Detecta el color del semáforo (rojo, amarillo, verde) en la parte
# superior-central de la imagen usando OpenCV (HSV + máscaras de color).
#
# ROI: franja superior central del frame
#   - Vertical  : 0 .. ROI_H_FRAC  (fracción superior, ej. 30 %)
#   - Horizontal: centro ± ROI_W_HALF_FRAC (ej. ±25 % → 50 % central)
#
# TÓPICOS
# ────────
#   Sub : /image/raw          [sensor_msgs/Image]   (desde camera_node)
#   Pub : /semaforo/estado    [std_msgs/String]      ("rojo","amarillo","verde","ninguno")
#   Pub : /semaforo/debug_img [sensor_msgs/Image]    (frame con ROI anotado)
# =============================================================================

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS DE ROI
# ─────────────────────────────────────────────────────────────────────────────
ROI_H_FRAC      = 0.30   # fracción superior del frame (0–30 %)
ROI_W_HALF_FRAC = 0.25   # mitad del ancho central (±25 % → 50 % del ancho)

# ─────────────────────────────────────────────────────────────────────────────
# RANGOS HSV DE COLORES
# ─────────────────────────────────────────────────────────────────────────────
# Rojo aparece en dos rangos en HSV (cruza el 0/180)
RED_LO1  = np.array([  0, 100,  80])
RED_HI1  = np.array([ 10, 255, 255])
RED_LO2  = np.array([165, 100,  80])
RED_HI2  = np.array([180, 255, 255])

YELLOW_LO = np.array([ 18, 100,  80])
YELLOW_HI = np.array([ 35, 255, 255])

GREEN_LO  = np.array([ 40, 80,   60])
GREEN_HI  = np.array([ 90, 255, 255])

# Mínimo de píxeles de color para considerar detección válida
MIN_PIXELS = 150


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

        self._bridge = CvBridge()

        # Publishers
        self._pub_estado = self.create_publisher(String, '/semaforo/estado',    10)
        self._pub_debug  = self.create_publisher(Image,  '/semaforo/debug_img', 10)

        # Subscriber
        self.create_subscription(Image, '/image/raw', self._image_cb, qos_be)

        self.get_logger().info(
            f'SemaforoNode listo | ROI superior {int(ROI_H_FRAC*100)}% '
            f'x central {int(ROI_W_HALF_FRAC*200)}%'
        )

    # ── Callback principal ────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        estado, debug_frame = self._detect(frame)

        # Publicar estado
        state_msg = String()
        state_msg.data = estado
        self._pub_estado.publish(state_msg)

        # Publicar debug
        dbg_msg = self._bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
        dbg_msg.header = msg.header
        self._pub_debug.publish(dbg_msg)

    # ── Detección de color en ROI ─────────────────────────────────────────

    def _detect(self, frame: np.ndarray) -> tuple:
        """
        Analiza la ROI superior-central del frame.
        Devuelve (estado: str, debug_frame: ndarray).
        """
        h, w = frame.shape[:2]

        # Calcular coordenadas del ROI
        roi_y0 = 0
        roi_y1 = int(h * ROI_H_FRAC)
        cx     = w // 2
        half_w = int(w * ROI_W_HALF_FRAC)
        roi_x0 = max(0, cx - half_w)
        roi_x1 = min(w, cx + half_w)

        roi = frame[roi_y0:roi_y1, roi_x0:roi_x1]

        # Suavizado para reducir ruido
        blurred = cv2.GaussianBlur(roi, (7, 7), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Máscaras
        mask_red    = (cv2.inRange(hsv, RED_LO1,  RED_HI1) |
                       cv2.inRange(hsv, RED_LO2,  RED_HI2))
        mask_yellow = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
        mask_green  = cv2.inRange(hsv, GREEN_LO,  GREEN_HI)

        px_red    = int(cv2.countNonZero(mask_red))
        px_yellow = int(cv2.countNonZero(mask_yellow))
        px_green  = int(cv2.countNonZero(mask_green))

        # Determinar estado (prioridad: rojo > amarillo > verde)
        if px_red >= MIN_PIXELS and px_red >= px_yellow and px_red >= px_green:
            estado     = 'rojo'
            box_color  = (0, 0, 220)
        elif px_yellow >= MIN_PIXELS and px_yellow >= px_green:
            estado     = 'amarillo'
            box_color  = (0, 200, 220)
        elif px_green >= MIN_PIXELS:
            estado     = 'verde'
            box_color  = (0, 200, 60)
        else:
            estado     = 'ninguno'
            box_color  = (120, 120, 120)

        # ── Frame de depuración ───────────────────────────────────────
        debug = frame.copy()

        # Rectángulo del ROI
        cv2.rectangle(debug, (roi_x0, roi_y0), (roi_x1, roi_y1), box_color, 2)

        # Texto de estado sobre el ROI
        label = f'SEMAFORO: {estado.upper()} | R={px_red} A={px_yellow} V={px_green}'
        cv2.putText(debug, label,
                    (roi_x0, max(roi_y1 + 18, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Overlay de máscara ganadora dentro del ROI
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

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
