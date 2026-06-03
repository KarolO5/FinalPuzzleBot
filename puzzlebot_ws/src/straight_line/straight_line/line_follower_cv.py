#!/usr/bin/env python3
# =============================================================================
# line_follower_cv.py
# =============================================================================
# Seguidor de línea negra con OpenCV + controlador PD.
# Integra estado del semáforo y señales de tránsito.
#
# ── MÁQUINA DE ESTADOS ────────────────────────────────────────────────────────
#
#  FOLLOWING ──red light──► STOPPED_LIGHT (vel=0, espera verde)
#  FOLLOWING ──STOP sign──► STOPPED_SIGN  (vel=0, 3 s, cooldown 10 s)
#  FOLLOWING ──intersec+TurnR──► TURNING  (gira 90° con odometría)
#  FOLLOWING ──intersec+TurnL──► TURNING  (gira -90° con odometría)
#  FOLLOWING ──intersec+AOnly──► STRAIGHT_OVERRIDE (avanza recto N frames)
#  Crossing visible       ──► vel_factor = 0.5
#  Give activo            ──► vel_factor = 0.5 hasta próxima intersección
#
# ── INTERSECCIÓN ──────────────────────────────────────────────────────────────
# Se detecta cuando ≥ INTERSECTION_MIN_COLS columnas del detector están activas
# simultáneamente (la línea se ensancha en "T" o cruce).
# Se dispara solo en la transición False→True para evitar activaciones repetidas.
#
# ── EVIDENCIA ────────────────────────────────────────────────────────────────
# • Frame + máscara cada EVIDENCE_INTERVAL_S segundos.
# • Mantiene solo MAX_EVIDENCE archivos por carpeta (rota el más antiguo).
#
# TÓPICOS
# ────────
#   Sub : /image/raw         [sensor_msgs/Image]
#   Sub : /odom              [nav_msgs/Odometry]
#   Sub : /semaforo/estado   [std_msgs/String]
#   Sub : /sign/state        [std_msgs/String]
#   Pub : /cmd_vel           [geometry_msgs/Twist]
#   Pub : /vision/debug_img  [sensor_msgs/Image]
#   Pub : /vision/error      [std_msgs/Float32]
# =============================================================================

import math
import os
import glob
import time
from enum import Enum

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
# PARÁMETROS DE ROBOT
# ─────────────────────────────────────────────────────────────────────────────
WHEEL_RADIUS = 0.0525
WHEEL_BASE   = 0.164
MAX_LINEAR   = 0.20
MAX_ANGULAR  = 0.35
LINEAR_VEL   = 0.15

# PD visual
KP_VIS = 1.8
KD_VIS = 0.25

# Visión — ROI
ROI_FRACTION  = 0.40
ROI_LEFT      = 0.30
ROI_RIGHT     = 0.70
N_COLS        = 8
BLUR_K        = 5
THRESH_VAL    = 60
MIN_CELL_FILL = 0.05

# Recovery
RECOVERY_FRAMES = 20
RECOVERY_OMEGA  = 0.25

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS DE SEÑALES
# ─────────────────────────────────────────────────────────────────────────────
INTERSECTION_MIN_COLS = 6     # columnas activas para detectar intersección
STOP_SIGN_DURATION    = 3.0   # segundos detenido ante señal STOP
STOP_SIGN_COOLDOWN    = 10.0  # cooldown entre activaciones de STOP
CROSSING_VEL_FACTOR   = 0.5   # factor de velocidad para Crossing
GIVE_VEL_FACTOR       = 0.5   # factor de velocidad para Give
TURN_ANGLE            = math.pi / 2.0   # 90 grados
TURN_OMEGA            = 0.28            # rad/s durante giro
TURN_ANGLE_TOL        = 0.08            # tolerancia angular ~4.6°
STRAIGHT_OVERRIDE_FRAMES = 30           # frames de avance recto en AOnly

# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCIA
# ─────────────────────────────────────────────────────────────────────────────
EVIDENCE_DIR         = os.path.expanduser('~/puzzlebot_evidence/line_follower')
FRAMES_DIR           = os.path.join(EVIDENCE_DIR, 'frames')
MASKS_DIR            = os.path.join(EVIDENCE_DIR, 'masks')
EVIDENCE_INTERVAL_S  = 5.0
MAX_EVIDENCE         = 5


def _save_evidence_pair(frame: np.ndarray, mask: np.ndarray) -> None:
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR,  exist_ok=True)
    ts = int(time.time() * 1000)
    cv2.imwrite(os.path.join(FRAMES_DIR, f'frame_{ts}.jpg'), frame)
    cv2.imwrite(os.path.join(MASKS_DIR,  f'mask_{ts}.jpg'),  mask)
    _rotate(FRAMES_DIR, '*.jpg')
    _rotate(MASKS_DIR,  '*.jpg')


def _rotate(directory: str, pattern: str) -> None:
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    while len(files) > MAX_EVIDENCE:
        os.remove(files.pop(0))


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(target: float, current: float) -> float:
    """Diferencia angular con wrapping a [-π, π]."""
    d = target - current
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d


# ─────────────────────────────────────────────────────────────────────────────
# ESTADO DEL NODO
# ─────────────────────────────────────────────────────────────────────────────

class RobotState(Enum):
    FOLLOWING         = 'following'
    STOPPED_LIGHT     = 'stopped_light'    # semáforo rojo
    STOPPED_SIGN      = 'stopped_sign'     # señal STOP
    TURNING           = 'turning'          # giro en intersección
    STRAIGHT_OVERRIDE = 'straight_override'  # AOnly en intersección


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
        """
        Retorna (error_norm, found, debug_frame, densities, mask_full, at_intersection).
        mask_full: máscara binaria del tamaño del frame original (para evidencia).
        at_intersection: True si ≥ INTERSECTION_MIN_COLS columnas activas.
        """
        h, w = frame.shape[:2]

        roi_y0 = int(h * (1.0 - self.roi_frac))
        x0_roi = int(w * self.roi_left)
        x1_roi = int(w * self.roi_right)

        roi          = frame[roi_y0:h, x0_roi:x1_roi]
        roi_h, roi_w = roi.shape[:2]

        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)
        _, mask = cv2.threshold(blurred, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)

        # Máscara tamaño completo para evidencia
        mask_full = np.zeros((h, w), dtype=np.uint8)
        mask_full[roi_y0:h, x0_roi:x1_roi] = mask

        col_w     = roi_w / self.n_cols
        densities = np.zeros(self.n_cols, dtype=np.float32)
        cell_pxls = roi_h * col_w

        for i in range(self.n_cols):
            cx0          = int(i * col_w)
            cx1          = int((i + 1) * col_w)
            densities[i] = np.count_nonzero(mask[:, cx0:cx1]) / cell_pxls

        found  = densities.max() > MIN_CELL_FILL
        active = densities > MIN_CELL_FILL

        at_intersection = int(active.sum()) >= INTERSECTION_MIN_COLS

        if found and active.sum() > 0:
            col_centers = np.array([(i + 0.5) * col_w for i in range(self.n_cols)])
            weighted_cx = float(np.sum(col_centers[active] * densities[active]) /
                                np.sum(densities[active]))
        else:
            weighted_cx = roi_w / 2.0

        error_norm = (roi_w / 2.0 - weighted_cx) / (roi_w / 2.0)

        # ── Frame de depuración ─────────────────────────────────────────
        debug = frame.copy()

        overlay = debug.copy()
        cv2.rectangle(overlay, (0, roi_y0),      (x0_roi, h), (0, 0, 0), -1)
        cv2.rectangle(overlay, (x1_roi, roi_y0), (w, h),      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, debug, 0.5, 0, debug)

        cv2.line(debug, (x0_roi, roi_y0), (x1_roi, roi_y0), (0, 255, 255), 1)
        cv2.line(debug, (x0_roi, roi_y0), (x0_roi, h),      (0, 255, 255), 1)
        cv2.line(debug, (x1_roi, roi_y0), (x1_roi, h),      (0, 255, 255), 1)

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

        cx_frame  = x0_roi + int(weighted_cx)
        cx_center = (x0_roi + x1_roi) // 2

        if found:
            cv2.line(debug, (cx_frame, roi_y0), (cx_frame, h), (0, 0, 255), 2)
        cv2.line(debug, (cx_center, roi_y0), (cx_center, h), (255, 0, 0), 1)

        arrow_y = roi_y0 + roi_h // 2
        cv2.arrowedLine(debug, (cx_center, arrow_y), (cx_frame, arrow_y),
                        (0, 255, 255) if found else (0, 0, 100), 2, tipLength=0.3)

        if at_intersection:
            cv2.putText(debug, 'INTERSECCION', (x0_roi + 4, roi_y0 - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        state_txt = f'err={error_norm:+.3f}' if found else 'NO LINE'
        state_col = (0, 255, 100) if found else (0, 50, 255)
        cv2.putText(debug, state_txt, (x0_roi + 4, roi_y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_col, 2)

        return error_norm, found, debug, densities, mask_full, at_intersection


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

        # ── Estado PD ─────────────────────────────────────────────────────
        self._prev_error  = 0.0
        self._prev_time   = None
        self._last_error  = 0.0
        self._frames_lost = 0

        # ── Semáforo ───────────────────────────────────────────────────────
        self._semaforo = 'ninguno'

        # ── Odometría ─────────────────────────────────────────────────────
        self._current_yaw = 0.0
        self._odom_ready  = False

        # ── Máquina de estados ─────────────────────────────────────────────
        self._state         = RobotState.FOLLOWING
        self._state_timer   = 0.0    # timestamp cuando el estado comenzó
        self._turn_dir      = 0.0    # +1 = izquierda, -1 = derecha
        self._turn_start_yaw = 0.0
        self._straight_frames = 0

        # ── Estado de señales ─────────────────────────────────────────────
        self._current_sign       = 'ninguna'
        self._pending_turn       = None   # 'right', 'left', o None
        self._pending_straight   = False  # AOnly
        self._give_active        = False
        self._crossing_visible   = False
        self._stop_cooldown_until = 0.0   # timestamp hasta el que STOP está bloqueado

        # ── Intersección ──────────────────────────────────────────────────
        self._was_at_intersection = False

        # ── Evidencia ─────────────────────────────────────────────────────
        self._last_evidence_time = 0.0

        # ── Publishers ────────────────────────────────────────────────────
        self._pub_cmd = self.create_publisher(Twist,   '/cmd_vel',          qos_be)
        self._pub_dbg = self.create_publisher(Image,   '/vision/debug_img', 10)
        self._pub_err = self.create_publisher(Float32, '/vision/error',     10)

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(Image,    '/image/raw',       self._image_cb,    qos_be)
        self.create_subscription(Odometry, '/odom',            self._odom_cb,     qos_be)
        self.create_subscription(String,   '/semaforo/estado', self._semaforo_cb, 10)
        self.create_subscription(String,   '/sign/state',      self._sign_cb,     10)

        self.get_logger().info(
            f'LineFollowerCV listo\n'
            f'  KP={KP_VIS}  KD={KD_VIS}  v={LINEAR_VEL} m/s\n'
            f'  ROI inferior={int(ROI_FRACTION*100)}% | '
            f'zona activa={int((ROI_RIGHT-ROI_LEFT)*100)}% central | '
            f'{N_COLS} columnas\n'
            f'  Detección intersección ≥{INTERSECTION_MIN_COLS} cols activas'
        )

    # ── Callbacks de sensores ────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self._current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self._odom_ready  = True

    def _semaforo_cb(self, msg: String):
        nuevo = msg.data
        if nuevo != self._semaforo:
            self.get_logger().info(f'Semáforo: {self._semaforo} → {nuevo}')
        self._semaforo = nuevo

    def _sign_cb(self, msg: String):
        sign = msg.data
        if sign != self._current_sign:
            self.get_logger().info(f'Señal: {self._current_sign} → {sign}')

        self._current_sign    = sign
        self._crossing_visible = (sign == 'Crossing')

        now = time.time()

        # STOP: registrar orden solo si no está en cooldown y estamos en FOLLOWING
        if (sign == 'STOP'
                and self._state == RobotState.FOLLOWING
                and now >= self._stop_cooldown_until):
            self._state       = RobotState.STOPPED_SIGN
            self._state_timer = now
            self._stop_cooldown_until = now + STOP_SIGN_DURATION + STOP_SIGN_COOLDOWN
            self.get_logger().info('STOP: deteniendo robot 3 s')

        # Giros: registrar pendiente (se ejecutan en la próxima intersección)
        elif sign == 'TurnR' and self._pending_turn is None:
            self._pending_turn = 'right'
            self.get_logger().info('TurnR: pendiente en próxima intersección')

        elif sign == 'TurnL' and self._pending_turn is None:
            self._pending_turn = 'left'
            self.get_logger().info('TurnL: pendiente en próxima intersección')

        elif sign == 'AOnly' and not self._pending_straight:
            self._pending_straight = True
            self.get_logger().info('AOnly: recto en próxima intersección')

        elif sign == 'Give':
            self._give_active = True

    # ── Callback de imagen principal ─────────────────────────────────────────

    def _image_cb(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        result = self._detector.process(frame)
        error_norm, found, debug_frame, densities, mask_full, at_intersection = result

        # Evidencia periódica
        now = time.time()
        if now - self._last_evidence_time >= EVIDENCE_INTERVAL_S:
            _save_evidence_pair(frame, mask_full)
            self._last_evidence_time = now

        # Detectar FLANCO de entrada a intersección
        entered_intersection = at_intersection and not self._was_at_intersection
        self._was_at_intersection = at_intersection

        if entered_intersection:
            self._on_intersection_enter()

        # Si Give activo y salimos de intersección, limpiar
        if self._give_active and not at_intersection and self._was_at_intersection:
            self._give_active = False
            self.get_logger().info('Give: velocidad normal restaurada')

        dbg_msg        = self._bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
        dbg_msg.header = msg.header
        self._pub_dbg.publish(dbg_msg)

        err_msg      = Float32()
        err_msg.data = float(error_norm)
        self._pub_err.publish(err_msg)

        self._run_control(error_norm, found)

    # ── Manejo de intersecciones ─────────────────────────────────────────────

    def _on_intersection_enter(self):
        if self._state != RobotState.FOLLOWING:
            return  # no actuar si ya estamos en otro estado

        if self._pending_turn == 'right':
            self.get_logger().info('Intersección: giro DERECHA')
            self._start_turn(-1.0)
            self._pending_turn = None

        elif self._pending_turn == 'left':
            self.get_logger().info('Intersección: giro IZQUIERDA')
            self._start_turn(1.0)
            self._pending_turn = None

        elif self._pending_straight:
            self.get_logger().info('Intersección: AOnly - avance recto')
            self._state           = RobotState.STRAIGHT_OVERRIDE
            self._straight_frames = 0
            self._pending_straight = False

        # Give: limpiar al cruzar la intersección
        if self._give_active:
            self._give_active = False
            self.get_logger().info('Give: cruce completado, velocidad normal')

    def _start_turn(self, direction: float):
        """direction: +1 = izquierda, -1 = derecha."""
        if not self._odom_ready:
            self.get_logger().warn('Odometría no disponible, saltando giro')
            return
        self._state          = RobotState.TURNING
        self._turn_dir       = direction
        self._turn_start_yaw = self._current_yaw

    # ── Control principal ────────────────────────────────────────────────────

    def _run_control(self, error_norm: float, found: bool):
        now = time.time()
        cmd = Twist()

        # ── Semáforo rojo (máxima prioridad) ──────────────────────────────
        if self._semaforo == 'rojo':
            if self._state != RobotState.STOPPED_LIGHT:
                self._state = RobotState.STOPPED_LIGHT
            self._pub_cmd.publish(Twist())
            return

        # Si estábamos parados por luz roja y ya no hay rojo, retomar
        if self._state == RobotState.STOPPED_LIGHT:
            self._state = RobotState.FOLLOWING

        # ── STOP sign ─────────────────────────────────────────────────────
        if self._state == RobotState.STOPPED_SIGN:
            if now - self._state_timer >= STOP_SIGN_DURATION:
                self.get_logger().info('STOP: reanudando seguimiento de línea')
                self._state = RobotState.FOLLOWING
            else:
                self._pub_cmd.publish(Twist())
                return

        # ── Giro en intersección ───────────────────────────────────────────
        if self._state == RobotState.TURNING:
            target_yaw = self._turn_start_yaw + self._turn_dir * TURN_ANGLE
            diff       = angle_diff(target_yaw, self._current_yaw)

            if abs(diff) <= TURN_ANGLE_TOL:
                self.get_logger().info('Giro completado')
                self._state = RobotState.FOLLOWING
            else:
                cmd.linear.x  = 0.05   # avance lento durante el giro
                cmd.angular.z = self._turn_dir * TURN_OMEGA
                self._pub_cmd.publish(cmd)
            return

        # ── AOnly recto en intersección ───────────────────────────────────
        if self._state == RobotState.STRAIGHT_OVERRIDE:
            self._straight_frames += 1
            if self._straight_frames >= STRAIGHT_OVERRIDE_FRAMES:
                self._state = RobotState.FOLLOWING
            else:
                cmd.linear.x  = LINEAR_VEL
                cmd.angular.z = 0.0
                self._pub_cmd.publish(cmd)
            return

        # ── Seguimiento normal (FOLLOWING) ────────────────────────────────
        # Factor de velocidad por señales activas
        vel_factor = 1.0
        if self._semaforo == 'amarillo':
            vel_factor = 0.5
        elif self._crossing_visible:
            vel_factor = CROSSING_VEL_FACTOR
        elif self._give_active:
            vel_factor = GIVE_VEL_FACTOR

        self._run_pd(error_norm, found, vel_factor)

    def _run_pd(self, error_norm: float, found: bool, vel_factor: float):
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
