import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

WHEEL_RADIUS = 0.0525
WHEEL_BASE   = 0.164
MAX_LINEAR   = 0.15
MAX_ANGULAR  = 0.30
Kp = 1.2
Kd = 0.3
LINEAR_VEL = 0.12

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class PDController(Node):
    def __init__(self):
        super().__init__('straight_line_pd')

        qos_be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.pub = self.create_publisher(Twist, '/cmd_vel', qos_be)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        self.target_yaw  = None
        self.current_yaw = 0.0
        self.prev_error  = 0.0
        self.prev_time   = None
        self.odom_ready  = False

        self.create_timer(0.05, self.control_loop)
        self.get_logger().info(f'PD listo | Kp={Kp} Kd={Kd} v={LINEAR_VEL} m/s')

    def odom_cb(self, msg: Odometry):
        self.current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.odom_ready  = True
        if self.target_yaw is None:
            self.target_yaw = self.current_yaw
            self.get_logger().info(f'Heading objetivo: {math.degrees(self.target_yaw):.1f}°')

    def control_loop(self):
        if not self.odom_ready or self.target_yaw is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        error = self.target_yaw - self.current_yaw
        error = math.atan2(math.sin(error), math.cos(error))
        dt = (now - self.prev_time) if self.prev_time else 1e-3
        d_error = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        self.prev_time  = now
        angular = clamp(Kp * error + Kd * d_error, -MAX_ANGULAR, MAX_ANGULAR)
        cmd = Twist()
        cmd.linear.x  = LINEAR_VEL
        cmd.angular.z = angular
        self.pub.publish(cmd)

    def stop(self):
        self.pub.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = PDController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.get_logger().info('Motores detenidos.')
        node.destroy_node()
        rclpy.shutdown()