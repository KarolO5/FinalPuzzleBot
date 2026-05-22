#!/usr/bin/env python3
# =============================================================================
# line_follower_cv.py
# =============================================================================
# Seguidor de línea negra con OpenCV + controlador PD.
#
# ROI activa: 40% inferior del frame, ignorando 30% de cada lado lateral.
# Zona analizada = 40% central del ancho del frame.
#
#  ┌──────────────────────────────────────────┐
#  │              zona ignorada               │  60% superior
#  ├───────┬─────────────────┬────────────────┤
#  │  30%  │   ZONA ACTIVA   │      30%       │  40% inferior
#  │ ignor │   (N columnas)  │    ignorado    │
#  └───────┴─────────────────┴────────────────┘
#
# TÓPICOS
#   Sub : /image/raw         [sensor_msgs/Image]
#   Sub : /odom              [nav_msgs/Odometry]
#   Sub : /semaforo/estado   [std_msgs/String]
#   Pub : /cmd_vel           [geometry_msgs/Twist]
#   Pub : /vision/debug_img  [sensor_msgs/Image]
#   Pub : /vision/error      [std_msgs/Float32]
# =============================================================================

import math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg       import Odometry
from sensor_msgs.msg    import Image
from std_msgs.msg       import Float32, String
from cv_bridge          import CvBridge
from rclpy.qos          import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────

# Robot
WHEEL_RADIUS = 0.0525
WHEEL_BASE   = 0.164
MAX_LINEAR   = 0.20
MAX_ANGULAR  = 0.35
LINEAR_VEL   = 0.15

# PD visual
KP_VIS = 1.8
KD_VIS = 0.25

# Visión — ROI
ROI_FRACTION  = 0.40   # Fracción inferior del frame (alto)
ROI_LEFT      = 0.30   # Ignorar 30% desde el borde izquierdo
ROI_RIGHT     = 0.70   # Ignorar 30% desde el borde derecho
N_COLS        = 8      # Columnas en la zona activa
BLUR_K        = 5      # Kernel Gaussian Blur (impar)
THRESH_VAL    = 60     # Umbral binario inverso
MIN_CELL_FILL = 0.05   # Densidad mínima para celda activa

# Recovery
RECOVERY_FRAMES = 20
RECOVERY_OMEGA  = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


# ─────────────────────────────────────────────────────────────────────────────
# DETECCIÓN POR GRILLA
# ─────────────────────────────────────────────────────────────────────────────

class GridLineDetector:

    def __init__(self, n_cols: int = N_COLS,
                 roi_frac: float = ROI_FRACTION,
                 roi_left: float = ROI_LEFT,
                 roi_right: float = ROI_RIGHT):
        self.n_cols    = n_cols
        self.roi_frac  = roi_frac
        self.roi_left  = roi_left
        self.roi_right = roi_right

    def process(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        # ── 1. Recortar ROI: zona inferior + zona central lateral ──────
        roi_y0 = int(h * (1.0 - self.roi_frac))
        x0_roi = int(w * self.roi_left)
        x1_roi = int(w * self.roi_right)

        roi           = frame[roi_y0:h, x0_roi:x1_roi]
        roi_h, roi_w  = roi.shape[:2]

        # ── 2. Pipeline de umbralización ──────────────────────────────
        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)
        _, mask = cv2.threshold(blurred, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)

        # ── 3. Grilla de N_COLS columnas ───────────────────────────────
        col_w     = roi_w / self.n_cols
        densities = np.zeros(self.n_cols, dtype=np.float32)
        cell_pxls = roi_h * col_w

        for i in range(self.n_cols):
            cx0           = int(i * col_w)
            cx1           = int((i + 1) * col_w)
            white_pixels  = np.count_nonzero(mask[:, cx0:cx1])
            densities[i]  = white_pixels / cell_pxls

        # ── 4. Centroide ponderado ─────────────────────────────────────
        found  = densities.max() > MIN_CELL_FILL
        active = densities > MIN_CELL_FILL

        if found and active.sum() > 0:
            col_centers = np.array([(i + 0.5) * col_w for i in range(self.n_cols)])
            weighted_cx = float(np.sum(col_centers[active] * densities[active]) /
                                np.sum(densities[active]))
        else:
            weighted_cx = roi_w / 2.0

        # Error normalizado respecto al centro de la zona activa
        error_norm = (roi_w / 2.0 - weighted_cx) / (roi_w / 2.0)

        # ── 5. Frame de depuración ─────────────────────────────────────
        debug = frame.copy()

        # Oscurecer zonas ignoradas izquierda y derecha
        overlay = debug.copy()
        cv2.rectangle(overlay, (0, roi_y0),      (x0_roi, h), (0, 0, 0), -1)
        cv2.rectangle(overlay, (x1_roi, roi_y0), (w, h),      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, debug, 0.5, 0, debug)

        # Bordes de la zona activa
        cv2.line(debug, (x0_roi, roi_y0), (x1_roi, roi_y0), (0, 255, 255), 1)
        cv2.line(debug, (x0_roi, roi_y0), (x0_roi, h),      (0, 255, 255), 1)
        cv2.line(debug, (x1_roi, roi_y0), (x1_roi, h),      (0, 255, 255), 1)

        # Dibujar celdas
        for i in range(self.n_cols):
            cx0  = x0_roi + int(i * col_w)
            cx1  = x0_roi + int((i + 1) * col_w)
            dens = densities[i]

            if dens > MIN_CELL_FILL:
                intensity    = int(clamp(dens * 3.0, 0.0, 1.0) * 255)
                cell_overlay = debug.copy()
                cv2.rectangle(cell_overlay, (cx0, roi_y0), (cx1, h),
                              (0, intensity, 0), -1)
                cv2.addWeighted(cell_overlay, 0.4, debug, 0.6, 0, debug)

            cv2.rectangle(debug, (cx0, roi_y0), (cx1, h), (80, 80, 80), 1)
            cv2.putText(debug, f'{dens:.2f}', (cx0 + 3, roi_y0 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (255, 255, 0) if dens > MIN_CELL_FILL else (80, 80, 80), 1)

        # Centroide
        cx_frame  = x0_roi + int(weighted_cx)
        cx_center = (x0_roi + x1_roi) // 2

        if found:
            cv2.line(debug, (cx_frame, roi_y0), (cx_frame, h), (0, 0, 255), 2)

        # Centro de la zona activa
        cv2.line(debug, (cx_center, roi_y0), (cx_center, h), (255, 0, 0), 1)

        # Flecha del error
        arrow_y = roi_y0 + roi_h // 2
        cv2.arrowedLine(debug, (cx_center, arrow_y), (cx_frame, arrow_y),
                        (0, 255, 255) if found else (0, 0, 100), 2, tipLength=0.3)

        # Texto de estado
        state_txt = f'err={error_norm:+.3f}' if found else 'NO LINE'
        state_col = (0, 255, 100) if found else (0, 50, 255)
        cv2.putText(debug, state_txt, (x0_roi + 4, roi_y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_col, 2)

        return error_norm, found, debug, densities


# ─────────────────────────────────────────────────────────────────────────────
# NODO ROS 2
# ─────────────────────────────────────────────────────────────────────────────

class LineFollowerCV(Node):

    def __init__(self):
        super().__init__('line_follower_cv')

        qos_be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._detector = GridLineDetector()
        self._bridge   = CvBridge()

        # Estado PD
        self._prev_error  = 0.0
        self._prev_time   = None
        self._last_error  = 0.0
        self._frames_lost = 0

        # Semáforo
        self._semaforo = 'ninguno'

        # Odometría
        self._current_yaw = 0.0
        self._odom_ready  = False

        # Publishers
        self._pub_cmd = self.create_publisher(Twist,   '/cmd_vel',          qos_be)
        self._pub_dbg = self.create_publisher(Image,   '/vision/debug_img', 10)
        self._pub_err = self.create_publisher(Float32, '/vision/error',     10)

        # Subscribers
        self.create_subscription(Image,    '/image/raw',       self._image_cb,    qos_be)
        self.create_subscription(Odometry, '/odom',            self._odom_cb,     qos_be)
        self.create_subscription(String,   '/semaforo/estado', self._semaforo_cb, 10)

        self.get_logger().info(
            f'LineFollowerCV listo\n'
            f'  KP={KP_VIS}  KD={KD_VIS}  v={LINEAR_VEL} m/s\n'
            f'  ROI inferior={int(ROI_FRACTION*100)}% | '
            f'zona activa={int((ROI_RIGHT-ROI_LEFT)*100)}% central | '
            f'{N_COLS} columnas'
        )

    def _odom_cb(self, msg: Odometry):
        self._current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self._odom_ready  = True

    def _semaforo_cb(self, msg: String):
        nuevo = msg.data
        if nuevo != self._semaforo:
            self.get_logger().info(f'Semáforo: {self._semaforo} → {nuevo}')
        self._semaforo = nuevo

    def _image_cb(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        error_norm, found, debug_frame, densities = self._detector.process(frame)

        dbg_msg        = self._bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
        dbg_msg.header = msg.header
        self._pub_dbg.publish(dbg_msg)

        err_msg      = Float32()
        err_msg.data = float(error_norm)
        self._pub_err.publish(err_msg)

        self._run_pd(error_norm, found)

    def _run_pd(self, error_norm: float, found: bool):
        # Semáforo rojo → detener
        if self._semaforo == 'rojo':
            self._pub_cmd.publish(Twist())
            return

        vel_factor = 0.5 if self._semaforo == 'amarillo' else 1.0

        now = self.get_clock().now().nanoseconds * 1e-9
        dt  = (now - self._prev_time) if self._prev_time is not None else 0.02
        dt  = max(dt, 1e-4)

        cmd = Twist()

        if found:
            self._frames_lost = 0
            self._last_error  = error_norm

            d_error = (error_norm - self._prev_error) / dt
            u       = KP_VIS * error_norm + KD_VIS * d_error
            u       = clamp(u, -MAX_ANGULAR, MAX_ANGULAR)

            cmd.linear.x  = LINEAR_VEL * vel_factor
            cmd.angular.z = u

        else:
            self._frames_lost += 1

            if self._frames_lost < RECOVERY_FRAMES:
                u             = KP_VIS * self._last_error * 0.5
                cmd.linear.x  = LINEAR_VEL * vel_factor * 0.5
                cmd.angular.z = clamp(u, -MAX_ANGULAR, MAX_ANGULAR)
            else:
                self.get_logger().warn(
                    f'Línea perdida {self._frames_lost} frames — buscando'
                )
                cmd.linear.x  = 0.0
                sign          = 1.0 if self._last_error >= 0 else -1.0
                cmd.angular.z = sign * RECOVERY_OMEGA

        self._prev_error = error_norm if found else self._prev_error
        self._prev_time  = now

        self._pub_cmd.publish(cmd)

    def stop(self):
        self._pub_cmd.publish(Twist())


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerCV()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.get_logger().info('Motores detenidos.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()