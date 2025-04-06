#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import math
import time


class YOLOTrackingPublisher(Node):
    def __init__(self):
        super().__init__('yolo_tracking_publisher')

        # 퍼블리셔 설정
        self.image_publisher = self.create_publisher(CompressedImage, 'AMR_image', 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # 서브스크라이버 설정
        self.goal_status_subscriber = self.create_subscription(String, '/goal_status', self.goal_status_callback, 10)

        # 유틸리티 설정
        self.bridge = CvBridge()
        self.model = YOLO('/home/rokey6/C2_py/best.pt')
        self.cap = cv2.VideoCapture(0)

        self.timer = self.create_timer(0.1, self.timer_callback)

        # 추적 상태 관리
        self.target_id = None  # 추적 중인 객체 ID
        self.frame_center = 320  # 프레임 중심 (640x480 해상도 기준)
        self.kid_detected = False  # 아이 감지 상태

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("웹캠에서 프레임을 가져오지 못했습니다.")
            return

        # YOLO로 객체 감지 수행
        results = self.model(frame, stream=True)
        kids = []
        parents = []

        # YOLO 감지 결과 처리
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
                confidence = math.ceil(box.conf[0] * 100) / 100  # 신뢰도 계산
                cls = int(box.cls[0])  # 클래스 ID

                if cls == 0:  # 'kid' 객체
                    kids.append((x1, y1, x2, y2, confidence))
                elif cls == 2:  # 'parent' 객체
                    parents.append((x1, y1, x2, y2))

        # 부모와 겹치지 않는 아이 선택
        selected_kid = None
        for kid in kids:
            x1, y1, x2, y2, confidence = kid
            overlaps = any(
                not (x2 < px1 or x1 > px2 or y2 < py1 or y1 > py2) for px1, py1, px2, py2 in parents
            )
            if not overlaps:
                selected_kid = kid
                break

        if selected_kid:
            # 선택된 아이 추적
            x1, y1, x2, y2, confidence = selected_kid
            self.get_logger().info(f"아이 추적 중: 중심=({(x1 + x2) // 2}, {(y1 + y2) // 2}), ID=1")
            self.follow_target((x1, y1, x2, y2))
        else:
            self.stop_motion()

        # 압축된 이미지를 퍼블리시
        _, compressed_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        compressed_img_msg = CompressedImage()
        compressed_img_msg.header.stamp = self.get_clock().now().to_msg()
        compressed_img_msg.format = "jpeg"
        compressed_img_msg.data = compressed_frame.tobytes()
        self.image_publisher.publish(compressed_img_msg)

    def follow_target(self, box):
        x1, y1, x2, y2 = box
        target_center_x = (x1 + x2) // 2
        target_width = x2 - x1

        twist = Twist()

        # 대상과의 거리 기반 선속도 조정
        if target_width < 80:
            twist.linear.x = 0.4
        elif target_width < 200:
            twist.linear.x = 0.2
        elif target_width > 270:
            twist.linear.x = -0.1
        else:
            twist.linear.x = 0.0

        # 대상의 수평 위치 기반 각속도 조정
        if target_center_x < self.frame_center - 30:
            twist.angular.z = -0.15
        elif target_center_x > self.frame_center + 30:
            twist.angular.z = 0.15
        else:
            twist.angular.z = 0.0

        self.cmd_vel_publisher.publish(twist)
        self.get_logger().info(f"아이를 따라가는 중: 선속도={twist.linear.x}, 각속도={twist.angular.z}")

    def stop_motion(self):
        twist = Twist()
        self.cmd_vel_publisher.publish(twist)
        self.get_logger().info("로봇 정지")

    def goal_status_callback(self, msg):
        self.get_logger().info(f"목표 상태: {msg.data}")

    def send_goal(self, x, y, yaw):
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        self.goal_publisher.publish(goal)
        self.get_logger().info(f"목표 전송: x={x}, y={y}, yaw={yaw}")

    def destroy_node(self):
        super().destroy_node()
        self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = YOLOTrackingPublisher()
    try:
        node.send_goal(2.0, 3.0, 1.57)  # 목표 위치 전송
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
