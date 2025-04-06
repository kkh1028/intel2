#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import math
import time

class YOLOTrackingPublisher(Node):
    def __init__(self):
        super().__init__('yolo_tracking_publisher')
        self.publisher_ = self.create_publisher(CompressedImage, 'AMR_image', 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        self.model = YOLO('/home/rokey6/C2_py/best.pt')
        self.cap = cv2.VideoCapture(0)
        self.timer = self.create_timer(0.1, self.timer_callback)

        # 추적 상태 관리
        self.target_id = None  # 추적 중인 객체 ID
        self.start_time_kid = None  # kid 탐지 시간
        self.start_time_parent = None  # parent 탐지 시간
        self.frame_center = 320  # Assuming 640x480 resolution

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame from webcam.")
            return

        results = self.model(frame, stream=True)

        kid_detected = False
        parent_detected = False
        kid_box = None
        current_time = time.time()

        # Iterate over results to extract detections
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])

                # Detect kid
                if cls == 0:  # Assuming 0 is 'kid'
                    kid_detected = True
                    kid_box = (x1, y1, x2, y2)

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f'Kid: {confidence:.2f}'
                    cv2.putText(frame, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Detect parent
                if cls == 2:  # Assuming 2 : 'parent'
                    parent_detected = True

        # Handle kid detection timing
        if kid_detected:
            if self.start_time_kid is None:
                self.start_time_kid = current_time
            elif current_time - self.start_time_kid >= 5.0 and self.target_id is None:
                self.get_logger().info("Starting to track the kid.")
                self.target_id = 1  # Start tracking the first kid
        else:
            self.start_time_kid = None

        # Handle parent detection timing
        if parent_detected:
            if self.start_time_parent is None:
                self.start_time_parent = current_time
            elif current_time - self.start_time_parent >= 5.0:
                self.get_logger().info("Parent detected for 5 seconds. Stopping tracking.")
                self.stop_motion()
                self.target_id = None  # Stop tracking
                return
        else:
            self.start_time_parent = None

        # Follow the kid if tracking is active
        if self.target_id is not None and kid_box:
            self.follow_target(kid_box)

        # Compress the frame and publish it as a ROS2 CompressedImage message
        _, compressed_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        compressed_img_msg = CompressedImage()
        compressed_img_msg.header.stamp = self.get_clock().now().to_msg()
        compressed_img_msg.format = "jpeg"
        compressed_img_msg.data = compressed_frame.tobytes()
        self.publisher_.publish(compressed_img_msg)

    def follow_target(self, box):
        x1, y1, x2, y2 = box
        target_center_x = (x1 + x2) // 2
        target_width = x2 - x1  # Bounding box width as an approximation of distance

        twist = Twist()

        # Adjust linear speed based on the size of the target box
        if target_width < 150:  # Target is far, move forward faster
            twist.linear.x = 0.4
        elif target_width < 250:  # Target is moderately far, move forward slowly
            twist.linear.x = 0.2
        elif target_width > 300:  # Target is too close, move backward
            twist.linear.x = -0.1
        else:  # Target is at the desired distance
            twist.linear.x = 0.0

        # Adjust angular speed based on horizontal position
        if target_center_x < self.frame_center - 30:  # Target is to the right
            twist.angular.z = -0.15
        elif target_center_x > self.frame_center + 30:  # Target is to the left
            twist.angular.z = 0.15
        else:  # Target is centered
            twist.angular.z = 0.0

        self.cmd_vel_publisher.publish(twist)
        self.get_logger().info(f"Following target: linear.x={twist.linear.x}, angular.z={twist.angular.z}")

    def stop_motion(self):
        twist = Twist()
        self.cmd_vel_publisher.publish(twist)
        self.get_logger().info("Stopping motion")

    def destroy_node(self):
        super().destroy_node()
        self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = YOLOTrackingPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
