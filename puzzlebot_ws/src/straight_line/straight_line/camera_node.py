#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('camera_index', 0)
        self.declare_parameter('fps', 30)
        idx = self.get_parameter('camera_index').value
        fps = self.get_parameter('fps').value

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1   # depth=1 es clave: los suscriptores siempre reciben el frame MÁS RECIENTE
        )

        self._bridge = CvBridge()
        self._pub = self.create_publisher(Image, '/image/raw', qos)

        self._cap = cv2.VideoCapture(idx)
        if not self._cap.isOpened():
            self.get_logger().error(f'No se pudo abrir cámara índice {idx}')
            raise RuntimeError('Cámara no disponible')

        period = 1.0 / fps
        self.create_timer(period, self._capture)
        self.get_logger().info(f'CameraNode listo | índice={idx} fps={fps}')

    def _capture(self):
        ret, frame = self._cap.read()
        if not ret:
            self.get_logger().warn('Frame vacío')
            return
        msg = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        self._pub.publish(msg)

    def destroy_node(self):
        if hasattr(self, '_cap') and self._cap.isOpened():
            self._cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()