import rclpy
from rclpy.node import Node
import math
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')

        self.declare_parameter('wheel_radius', 0.0525)
        self.declare_parameter('wheel_base',   0.164)

        self.r  = self.get_parameter('wheel_radius').value
        self.L  = self.get_parameter('wheel_base').value

        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0
        self.vL  = 0.0
        self.vR  = 0.0
        self.dt  = 0.05  # 20 Hz

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(Float32, '/VelocityEncL', self.vL_cb, qos)
        self.create_subscription(Float32, '/VelocityEncR', self.vR_cb, qos)
        self.pub_odom = self.create_publisher(Odometry, '/odom', 10)
        self.get_logger().info(f"[ODOM] x={self.x:.3f} y={self.y:.3f} yaw={math.degrees(self.yaw):.2f}")
        self.create_timer(self.dt, self.update)

        self.get_logger().info(f'Odometry listo | r={self.r} L={self.L}')

    def vL_cb(self, msg):
        self.vL = msg.data
        self.get_logger().info(f"vL: {self.vL}")

    def vR_cb(self, msg):
        self.vR = msg.data
        self.get_logger().info(f"vR: {self.vR}")
    def update(self):
        vl = self.vL * self.r
        vr = self.vR * self.r

        v = (vr + vl) / 2.0
        w = (vr - vl) / self.L

        self.x   += v * math.cos(self.yaw) * self.dt
        self.y   += v * math.sin(self.yaw) * self.dt
        self.yaw += w * self.dt

        while self.yaw >  math.pi: self.yaw -= 2 * math.pi
        while self.yaw < -math.pi: self.yaw += 2 * math.pi

        odom = Odometry()
        odom.header.stamp        = self.get_clock().now().to_msg()
        odom.header.frame_id     = 'odom'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.yaw / 2.0)
        odom.twist.twist.linear.x  = v
        odom.twist.twist.angular.z = w

        self.pub_odom.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
