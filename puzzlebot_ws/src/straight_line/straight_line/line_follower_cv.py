#!/usr/bin/env python3
# =============================================================================
# line_follower_cv.py
# =============================================================================
# Seguidor de línea negra con OpenCV + controlador PD existente.
#
# ESTRATEGIA DE DETECCIÓN — GRILLA EN LA FRANJA INFERIOR
# ──────────────────────────────────────────────────────────
# Se toma únicamente el 40 % inferior del frame (ROI horizontal).
# Esa franja se divide en N_COLS columnas (celdas) de igual ancho.
# En cada celda se cuenta la densidad de píxeles blancos (línea).
# La columna con mayor densidad determina la posición lateral de la línea.
# El error se calcula como la distancia del centro de esa columna al
# centro horizontal de la imagen → el PD corrige con esa señal.
#
# Esto es más robusto que solo el centroide global porque:
#   • Cada celda vota independientemente → menos sensible a ruido aislado.
#   • La resolución lateral es configurable (N_COLS).
#   • Fácil de visualizar en debug (se pinta cada celda).
#
# TÓPICOS
# ────────
#   Sub : /camera/image_raw  [sensor_msgs/Image]
#   Sub : /odom              [nav_msgs/Odometry]      (desde tu OdometryNode)
#   Pub : /cmd_vel           [geometry_msgs/Twist]
#   Pub : /vision/debug_img  [sensor_msgs/Image]      (frame anotado)
#   Pub : /vision/error      [std_msgs/Float32]       (error px normalizado)
# =============================================================================

import math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg       import Odometry
from sensor_msgs.msg    import Image
from std_msgs.msg       import Float32
from cv_bridge          import CvBridge
from rclpy.qos          import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS — ajusta estos sin tocar el código de control
# ─────────────────────────────────────────────────────────────────────────────

# Robot
WHEEL_RADIUS = 0.0525
WHEEL_BASE   = 0.164
MAX_LINEAR   = 0.20
MAX_ANGULAR  = 0.35
LINEAR_VEL   = 0.15      # m/s de crucero (un poco menos para tener margen de giro)

# PD visual — error en píxeles normalizados [-1, 1]
KP_VIS = 1.8             # Proporcional al error lateral de la línea
KD_VIS = 0.25            # Derivativo (amortigua oscilaciones)

# Visión
ROI_FRACTION  = 0.40     # Fracción inferior del frame que se analiza (0.4 = 40 %)
N_COLS        = 8        # Número de columnas (celdas) en la grilla
BLUR_K        = 5        # Tamaño del kernel Gaussian Blur (impar)
THRESH_VAL    = 60       # Umbral binario inverso (línea negra → blanco)
MIN_CELL_FILL = 0.05     # Fracción mínima de píxeles blancos para considerar celda activa

# Recovery (si se pierde la línea)
RECOVERY_FRAMES = 20     # Frames sin línea antes de girar en búsqueda
RECOVERY_OMEGA  = 0.25   # rad/s de giro durante la búsqueda


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
# PROCESAMIENTO DE IMAGEN — detección por grilla
# ─────────────────────────────────────────────────────────────────────────────

class GridLineDetector:
    """
    Detecta la línea negra dividiendo la franja inferior en N columnas.

    Retorna:
      error_norm  : float en [-1, 1]  (0 = centrada, neg = der, pos = izq)
      found       : bool
      debug_frame : imagen BGR anotada para publicar/mostrar
      densities   : array de densidades por celda (para debug)
    """

    def __init__(self, n_cols: int = N_COLS, roi_frac: float = ROI_FRACTION):
        self.n_cols   = n_cols
        self.roi_frac = roi_frac

    def process(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        # ── 1. Recortar ROI inferior ──────────────────────────────────
        roi_y0   = int(h * (1.0 - self.roi_frac))
        roi      = frame[roi_y0:h, :]           # shape: (roi_h, w, 3)
        roi_h, roi_w = roi.shape[:2]

        # ── 2. Pipeline de umbralización ─────────────────────────────
        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)
        # THRESH_BINARY_INV: píxeles oscuros (línea) → 255
        _, mask = cv2.threshold(blurred, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)

        # ── 3. Grilla de N_COLS columnas ──────────────────────────────
        col_w      = roi_w / self.n_cols
        densities  = np.zeros(self.n_cols, dtype=np.float32)
        cell_pxls  = roi_h * col_w             # píxeles totales por celda

        for i in range(self.n_cols):
            x0 = int(i * col_w)
            x1 = int((i + 1) * col_w)
            cell_mask    = mask[:, x0:x1]
            white_pixels = np.count_nonzero(cell_mask)
            densities[i] = white_pixels / cell_pxls   # fracción 0-1

        # ── 4. Selección de la columna dominante ──────────────────────
        best_col  = int(np.argmax(densities))
        best_dens = densities[best_col]
        found     = best_dens > MIN_CELL_FILL

        # Centroide ponderado entre columnas activas (más suave que argmax)
        active = densities > MIN_CELL_FILL
        if found and active.sum() > 0:
            col_centers = np.array([(i + 0.5) * col_w for i in range(self.n_cols)])
            weighted_cx = float(np.sum(col_centers[active] * densities[active]) /
                                np.sum(densities[active]))
        else:
            weighted_cx = roi_w / 2.0   # centro si no hay línea

        # Error normalizado: 0 = centrado, +1 = línea al extremo izq, -1 = der
        error_norm = (roi_w / 2.0 - weighted_cx) / (roi_w / 2.0)

        # ── 5. Frame de depuración ────────────────────────────────────
        debug = frame.copy()

        # Fondo semitransparente sobre la ROI
        overlay = debug.copy()
        cv2.rectangle(overlay, (0, roi_y0), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.25, debug, 0.75, 0, debug)

        # Dibujar borde de ROI
        cv2.line(debug, (0, roi_y0), (w, roi_y0), (0, 255, 255), 1)

        # Dibujar celdas
        for i in range(self.n_cols):
            x0 = int(i * col_w)
            x1 = int((i + 1) * col_w)
            y0 = roi_y0
            y1 = h
            dens = densities[i]

            if dens > MIN_CELL_FILL:
                # Intensidad del verde proporcional a la densidad
                intensity = int(clamp(dens * 3.0, 0.0, 1.0) * 255)
                color_fill = (0, intensity, 0)
                cv2.rectangle(debug, (x0, y0), (x1, y1), color_fill, -1)
                cv2.addWeighted(debug, 0.4, frame, 0.6, 0, debug)
            # Borde de celda
            cv2.rectangle(debug, (x0, y0), (x1, y1), (80, 80, 80), 1)
            # Densidad en texto
            cv2.putText(debug, f'{dens:.2f}', (x0 + 3, y0 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (255, 255, 0) if dens > MIN_CELL_FILL else (80, 80, 80), 1)

        # Línea de centroide detectado
        cx_frame = int(weighted_cx)
        if found:
            cv2.line(debug, (cx_frame, roi_y0), (cx_frame, h),
                     (0, 0, 255), 2)

        # Línea del centro de imagen
        cv2.line(debug, (w // 2, roi_y0), (w // 2, h), (255, 0, 0), 1)

        # Flecha del error
        arrow_y = roi_y0 + roi_h // 2
        cv2.arrowedLine(debug, (w // 2, arrow_y), (cx_frame, arrow_y),
                        (0, 255, 255) if found else (0, 0, 100), 2,
                        tipLength=0.3)

        # Texto de estado
        state_txt = f'err={error_norm:+.3f}' if found else 'NO LINE'
        state_col = (0, 255, 100) if found else (0, 50, 255)
        cv2.putText(debug, state_txt, (6, roi_y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_col, 2)

        return error_norm, found, debug, densities


# ─────────────────────────────────────────────────────────────────────────────
# NODO ROS 2
# ─────────────────────────────────────────────────────────────────────────────

class LineFollowerCV(Node):
    """
    Nodo seguidor de línea que combina:
      • Visión OpenCV con grilla de celdas
      • Controlador PD visual (error lateral de cámara)
      • Odometría de tu OdometryNode (usada para heading de respaldo)

    Flujo de control:
      image_raw → GridLineDetector → error_norm
                                          │
                                     PD visual
                                          │
                                      /cmd_vel
    """

    def __init__(self):
        super().__init__('line_follower_cv')

        # ── QoS ───────────────────────────────────────────────────────
        qos_be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Módulo de visión ──────────────────────────────────────────
        self._detector = GridLineDetector(n_cols=N_COLS, roi_frac=ROI_FRACTION)
        self._bridge   = CvBridge()

        # ── Estado PD ─────────────────────────────────────────────────
        self._prev_error  = 0.0
        self._prev_time   = None
        self._last_error  = 0.0       # último error válido (para recovery)
        self._frames_lost = 0         # frames consecutivos sin línea

        # ── Odometría (heredada de tu PD controller) ──────────────────
        self._current_yaw  = 0.0
        self._odom_ready   = False

        # ── Publishers ────────────────────────────────────────────────
        self._pub_cmd = self.create_publisher(Twist,   '/cmd_vel',          qos_be)
        self._pub_dbg = self.create_publisher(Image,   '/vision/debug_img', 10)
        self._pub_err = self.create_publisher(Float32, '/vision/error',     10)

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(
            Image, '/image/raw', self._image_cb, qos_be
        )
        self.create_subscription(
            Odometry, '/odom', self._odom_cb, qos_be
        )

        self.get_logger().info(
            f'LineFollowerCV listo\n'
            f'  KP={KP_VIS}  KD={KD_VIS}  v={LINEAR_VEL} m/s\n'
            f'  ROI={int(ROI_FRACTION*100)}% inferior | {N_COLS} columnas'
        )

    # ── Callback odometría ─────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self._current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self._odom_ready  = True

    # ── Callback imagen ────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        # Convertir a OpenCV
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        # Detectar línea con grilla
        error_norm, found, debug_frame, densities = self._detector.process(frame)

        # Publicar debug
        dbg_msg = self._bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
        dbg_msg.header = msg.header
        self._pub_dbg.publish(dbg_msg)

        # Publicar error
        err_msg      = Float32()
        err_msg.data = float(error_norm)
        self._pub_err.publish(err_msg)

        # Control
        self._run_pd(error_norm, found)

    # ── Controlador PD visual ──────────────────────────────────────────

    def _run_pd(self, error_norm: float, found: bool):
        """
        Controlador PD cuya señal de entrada es el error lateral normalizado
        de la cámara (igual que tu PDController usa el error de heading).

            u(t) = Kp·e + Kd·(de/dt)

        Las velocidades de rueda resultan:
            v_izq = v_base - u
            v_der = v_base + u
        → publicadas como Twist (linear.x, angular.z).
        """
        now = self.get_clock().now().nanoseconds * 1e-9
        dt  = (now - self._prev_time) if self._prev_time is not None else 0.02
        dt  = max(dt, 1e-4)

        cmd = Twist()

        if found:
            self._frames_lost = 0
            self._last_error  = error_norm

            # Derivada del error
            d_error = (error_norm - self._prev_error) / dt

            # Señal PD
            u = KP_VIS * error_norm + KD_VIS * d_error
            u = clamp(u, -MAX_ANGULAR, MAX_ANGULAR)

            cmd.linear.x  = LINEAR_VEL
            cmd.angular.z = u

            self.get_logger().debug(
                f'e={error_norm:+.3f}  de={d_error:+.4f}  u={u:+.3f}'
            )

        else:
            self._frames_lost += 1

            if self._frames_lost < RECOVERY_FRAMES:
                # Inercia: mantener último giro suavizado
                u = KP_VIS * self._last_error * 0.5
                cmd.linear.x  = LINEAR_VEL * 0.5
                cmd.angular.z = clamp(u, -MAX_ANGULAR, MAX_ANGULAR)
            else:
                # Búsqueda: girar hacia última dirección conocida
                self.get_logger().warn(
                    f'Línea perdida {self._frames_lost} frames — modo búsqueda'
                )
                cmd.linear.x  = 0.0
                sign = 1.0 if self._last_error >= 0 else -1.0
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