#!/usr/bin/env python3
# =============================================================================
# line_follower_cv.py
# =============================================================================
# Seguidor de línea negra con OpenCV + controlador PD.
#
# ── FLAGS DE MÓDULOS (activar/desactivar sin borrar código) ──────────────────
SEMAFORO_ENABLED = False   # True cuando semaforo.py esté en producción
SIGNS_ENABLED    = False   # True cuando sign_detector.py esté en producción
# ─────────────────────────────────────────────────────────────────────────────
#
# ── EVIDENCIA ────────────────────────────────────────────────────────────────
# Fotografía del frame cada EVIDENCE_INTERVAL_S segundos.
# Máximo MAX_EVIDENCE_TOTAL fotos. Cuando se supera ese límite se eliminan
# las MÁS RECIENTES (las últimas en tomarse), conservando siempre las primeras.
# Esto permite ver las condiciones iniciales del recorrido.
#
# TÓPICOS
# ────────
#   Sub : /image/raw         [sensor_msgs/Image]
#   Sub : /odom              [nav_msgs/Odometry]
#   Sub : /semaforo/estado   [std_msgs/String]   (solo si SEMAFORO_ENABLED)
#   Sub : /sign/state        [std_msgs/String]   (solo si SIGNS_ENABLED)
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
MAX_ANGULAR  = 0.20
LINEAR_VEL   = 0.15

# PD visual
KP_VIS = 0.9
KD_VIS = 0.40

# Filtro exponencial sobre el error (0 = sin filtro, valores altos = más suave)
# Sube ERROR_ALPHA si el robot sigue oscilando; bájalo si reacciona muy lento
ERROR_ALPHA = 0.4

# Visión — ROI
ROI_FRACTION  = 0.40
ROI_LEFT      = 0.30
ROI_RIGHT     = 0.70
N_COLS        = 8
BLUR_K        = 5
THRESH_VAL    = 60
MIN_CELL_FILL = 0.05

# Recovery (línea perdida)
RECOVERY_FRAMES = 20
RECOVERY_OMEGA  = 0.25

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS DE SEÑALES (solo usados si los flags están activos)
# ─────────────────────────────────────────────────────────────────────────────
INTERSECTION_MIN_COLS    = 6
STOP_SIGN_DURATION       = 3.0
STOP_SIGN_COOLDOWN       = 10.0
CROSSING_VEL_FACTOR      = 0.5
GIVE_VEL_FACTOR          = 0.5
TURN_ANGLE               = math.pi / 2.0
TURN_OMEGA               = 0.28
TURN_ANGLE_TOL           = 0.08
STRAIGHT_OVERRIDE_FRAMES = 30

# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCIA
# ─────────────────────────────────────────────────────────────────────────────
_HOME               = os.environ.get('HOME', '/tmp')
FRAMES_DIR          = os.path.join(_HOME, 'puzzlebot_evidence', 'line_follower', 'frames')
EVIDENCE_INTERVAL_S = 5.0      # una foto cada N segundos
MAX_EVIDENCE_TOTAL  = 10       # máximo de fotos almacenadas
# Cuando se supera MAX_EVIDENCE_TOTAL se eliminan las más nuevas (mayor timestamp),
# conservando las primeras fotos tomadas (menor timestamp).

try:
    os.makedirs(FRAMES_DIR, exist_ok=True)
except OSError as _e:
    import warnings
    warnings.warn(f'[line_follower] No se pudo crear directorio de evidencia: {_e}')


def _save_frame(frame: np.ndarray) -> None:
    """
    Guarda una foto. Si hay más de MAX_EVIDENCE_TOTAL archivos elimina los
    más recientes (mayor timestamp en el nombre), conservando los más antiguos.
    """
    try:
        ts   = int(time.time() * 1000)
        path = os.path.join(FRAMES_DIR, f'frame_{ts}.jpg')
        ok   = cv2.imwrite(path, frame)
        if not ok:
            import sys
            print(f'[evidencia] cv2.imwrite falló: {path}', file=sys.stderr)
            return

        # sorted() con nombres frame_TIMESTAMP.jpg ordena de menor a mayor timestamp
        # → files[0] = más antiguo, files[-1] = más reciente
        files = sorted(glob.glob(os.path.join(FRAMES_DIR, 'frame_*.jpg')))
        while len(files) > MAX_EVIDENCE_TOTAL:
            # Eliminar el MÁS RECIENTE (pop desde el final)
            os.remove(files.pop())

    except Exception as exc:
        import sys
        print(f'[evidencia] excepción: {exc}', file=sys.stderr)


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
    d = target - current
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d


# ─────────────────────────────────────────────────────────────────────────────
# ESTADO DEL NODO
# ─────────────────────────────────────────────────────────────────────────────

class RobotState(Enum):
    FOLLOWING         = 'following'
    STOPPED_LIGHT     = 'stopped_light'
    STOPPED_SIGN      = 'stopped_sign'
    TURNING           = 'turning'
    STRAIGHT_OVERRIDE = 'straight_override'


# ─────────────────────────────────────────────────────────────────────────────
# DETECCIÓN POR GRILLA
# ─────────────────────────────────────────────────────────────────────────────

class GridLineDetector:

    def __init__(self, n_cols=N_COLS, roi_frac=ROI_FRACTION,
                 roi_left=ROI_LEFT, roi_right=ROI_RIGHT):
        self.n_cols    = n_cols
        self.roi_frac  = roi_frac
        self.roi_left  = roi_left
        self.roi_right = roi_right

    def process(self, frame: np.ndarray):
        """
        Retorna (error_norm, found, debug_frame, densities, at_intersection).
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

        # ── Debug frame ────────────────────────────────────────────────────
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

        state_txt = f'err={error_norm:+.3f}' if found else 'NO LINE'
        state_col = (0, 255, 100) if found else (0, 50, 255)
        cv2.putText(debug, state_txt, (x0_roi + 4, roi_y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_col, 2)

        return error_norm, found, debug, densities, at_intersection


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
        self._prev_error     = 0.0
        self._filtered_error = 0.0
        self._prev_time      = None
        self._last_error     = 0.0
        self._frames_lost    = 0

        # ── Odometría ─────────────────────────────────────────────────────
        self._current_yaw = 0.0
        self._odom_ready  = False

        # ── Semáforo (inactivo si SEMAFORO_ENABLED = False) ───────────────
        self._semaforo = 'ninguno'

        # ── Señales (inactivo si SIGNS_ENABLED = False) ───────────────────
        self._state              = RobotState.FOLLOWING
        self._state_timer        = 0.0
        self._turn_dir           = 0.0
        self._turn_start_yaw     = 0.0
        self._straight_frames    = 0
        self._current_sign       = 'ninguna'
        self._pending_turn       = None
        self._pending_straight   = False
        self._give_active        = False
        self._crossing_visible   = False
        self._stop_cooldown_until = 0.0
        self._was_at_intersection = False

        # ── Evidencia ─────────────────────────────────────────────────────
        self._last_evidence_time = 0.0

        # ── Publishers ────────────────────────────────────────────────────
        self._pub_cmd = self.create_publisher(Twist,   '/cmd_vel',          qos_be)
        self._pub_dbg = self.create_publisher(Image,   '/vision/debug_img', 10)
        self._pub_err = self.create_publisher(Float32, '/vision/error',     10)

        # ── Subscribers siempre activos ───────────────────────────────────
        self.create_subscription(Image,    '/image/raw', self._image_cb, qos_be)
        self.create_subscription(Odometry, '/odom',      self._odom_cb,  qos_be)

        # ── Subscribers opcionales ────────────────────────────────────────
        if SEMAFORO_ENABLED:
            self.create_subscription(String, '/semaforo/estado',
                                     self._semaforo_cb, 10)
            self.get_logger().info('Semáforo: ACTIVO')
        else:
            self.get_logger().info('Semáforo: DESHABILITADO (SEMAFORO_ENABLED=False)')

        if SIGNS_ENABLED:
            self.create_subscription(String, '/sign/state',
                                     self._sign_cb, 10)
            self.get_logger().info('Señales: ACTIVO')
        else:
            self.get_logger().info('Señales: DESHABILITADO (SIGNS_ENABLED=False)')

        self.get_logger().info(
            f'LineFollowerCV listo\n'
            f'  KP={KP_VIS}  KD={KD_VIS}  alpha={ERROR_ALPHA}  v={LINEAR_VEL} m/s\n'
            f'  ROI inferior={int(ROI_FRACTION*100)}% | '
            f'zona activa={int((ROI_RIGHT-ROI_LEFT)*100)}% central | '
            f'{N_COLS} columnas\n'
            f'  Evidencia: {FRAMES_DIR}\n'
            f'  (foto cada {EVIDENCE_INTERVAL_S}s, máx {MAX_EVIDENCE_TOTAL}, '
            f'se conservan las primeras)'
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

        self._current_sign     = sign
        self._crossing_visible = (sign == 'Crossing')
        now = time.time()

        if (sign == 'STOP'
                and self._state == RobotState.FOLLOWING
                and now >= self._stop_cooldown_until):
            self._state       = RobotState.STOPPED_SIGN
            self._state_timer = now
            self._stop_cooldown_until = now + STOP_SIGN_DURATION + STOP_SIGN_COOLDOWN
            self.get_logger().info('STOP: deteniendo robot 3 s')
        elif sign == 'TurnR' and self._pending_turn is None:
            self._pending_turn = 'right'
        elif sign == 'TurnL' and self._pending_turn is None:
            self._pending_turn = 'left'
        elif sign == 'AOnly' and not self._pending_straight:
            self._pending_straight = True
        elif sign == 'Give':
            self._give_active = True

    # ── Callback de imagen ───────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        error_norm, found, debug_frame, densities, at_intersection = \
            self._detector.process(frame)

        # ── Evidencia ────────────────────────────────────────────────────
        now = time.time()
        if now - self._last_evidence_time >= EVIDENCE_INTERVAL_S:
            _save_frame(frame)
            self._last_evidence_time = now

        # ── Intersección (solo si señales activas) ────────────────────────
        if SIGNS_ENABLED:
            entered = at_intersection and not self._was_at_intersection
            self._was_at_intersection = at_intersection
            if entered:
                self._on_intersection_enter()
            if self._give_active and not at_intersection and self._was_at_intersection:
                self._give_active = False

        # ── Publicar debug y error ────────────────────────────────────────
        dbg_msg        = self._bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
        dbg_msg.header = msg.header
        self._pub_dbg.publish(dbg_msg)

        err_msg      = Float32()
        err_msg.data = float(error_norm)
        self._pub_err.publish(err_msg)

        self._run_control(error_norm, found)

    # ── Intersecciones ───────────────────────────────────────────────────────

    def _on_intersection_enter(self):
        if self._state != RobotState.FOLLOWING:
            return
        if self._pending_turn == 'right':
            self._start_turn(-1.0)
            self._pending_turn = None
        elif self._pending_turn == 'left':
            self._start_turn(1.0)
            self._pending_turn = None
        elif self._pending_straight:
            self._state           = RobotState.STRAIGHT_OVERRIDE
            self._straight_frames = 0
            self._pending_straight = False
        if self._give_active:
            self._give_active = False

    def _start_turn(self, direction: float):
        if not self._odom_ready:
            self.get_logger().warn('Odometría no disponible, saltando giro')
            return
        self._state          = RobotState.TURNING
        self._turn_dir       = direction
        self._turn_start_yaw = self._current_yaw

    # ── Control principal ────────────────────────────────────────────────────

    def _run_control(self, error_norm: float, found: bool):
        now = time.time()

        # Semáforo rojo (máxima prioridad, solo si activo)
        if SEMAFORO_ENABLED and self._semaforo == 'rojo':
            if self._state != RobotState.STOPPED_LIGHT:
                self._state = RobotState.STOPPED_LIGHT
            self._pub_cmd.publish(Twist())
            return
        if self._state == RobotState.STOPPED_LIGHT:
            self._state = RobotState.FOLLOWING

        # STOP sign
        if self._state == RobotState.STOPPED_SIGN:
            if now - self._state_timer >= STOP_SIGN_DURATION:
                self._state = RobotState.FOLLOWING
            else:
                self._pub_cmd.publish(Twist())
                return

        # Giro
        if self._state == RobotState.TURNING:
            target_yaw = self._turn_start_yaw + self._turn_dir * TURN_ANGLE
            diff       = angle_diff(target_yaw, self._current_yaw)
            if abs(diff) <= TURN_ANGLE_TOL:
                self._state = RobotState.FOLLOWING
            else:
                cmd = Twist()
                cmd.linear.x  = 0.05
                cmd.angular.z = self._turn_dir * TURN_OMEGA
                self._pub_cmd.publish(cmd)
            return

        # AOnly
        if self._state == RobotState.STRAIGHT_OVERRIDE:
            self._straight_frames += 1
            if self._straight_frames >= STRAIGHT_OVERRIDE_FRAMES:
                self._state = RobotState.FOLLOWING
            else:
                cmd = Twist()
                cmd.linear.x  = LINEAR_VEL
                cmd.angular.z = 0.0
                self._pub_cmd.publish(cmd)
            return

        # Factor de velocidad por señales activas
        vel_factor = 1.0
        if SEMAFORO_ENABLED and self._semaforo == 'amarillo':
            vel_factor = 0.5
        if SIGNS_ENABLED:
            if self._crossing_visible:
                vel_factor = CROSSING_VEL_FACTOR
            elif self._give_active:
                vel_factor = GIVE_VEL_FACTOR

        self._run_pd(error_norm, found, vel_factor)

    # ── Controlador PD con filtro exponencial ────────────────────────────────

    def _run_pd(self, error_norm: float, found: bool, vel_factor: float):
        now = self.get_clock().now().nanoseconds * 1e-9
        dt  = (now - self._prev_time) if self._prev_time is not None else 0.02
        dt  = max(dt, 1e-4)

        cmd = Twist()

        if found:
            self._frames_lost = 0
            self._last_error  = error_norm

            self._filtered_error = (ERROR_ALPHA * self._filtered_error +
                                    (1.0 - ERROR_ALPHA) * error_norm)

            d_error = (self._filtered_error - self._prev_error) / dt
            u       = KP_VIS * self._filtered_error + KD_VIS * d_error
            u       = clamp(u, -MAX_ANGULAR, MAX_ANGULAR)

            cmd.linear.x  = LINEAR_VEL * vel_factor
            cmd.angular.z = u

        else:
            self._frames_lost    += 1
            self._filtered_error *= 0.85

            if self._frames_lost < RECOVERY_FRAMES:
                u             = KP_VIS * self._last_error * 0.5
                cmd.linear.x  = LINEAR_VEL * vel_factor * 0.5
                cmd.angular.z = clamp(u, -MAX_ANGULAR, MAX_ANGULAR)
            else:
                self.get_logger().warn(
                    f'Línea perdida {self._frames_lost} frames — buscando')
                cmd.linear.x  = 0.0
                sign          = 1.0 if self._last_error >= 0 else -1.0
                cmd.angular.z = sign * RECOVERY_OMEGA

        self._prev_error = self._filtered_error if found else self._prev_error
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
