#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import threading
import time
import os
import math

class YoloPublisher(Node):
    def __init__(self):
        super().__init__('yolo_publisher')
        self.publisher_ = self.create_publisher(Image, 'processed_image', 10)
        self.bridge = CvBridge()
        self.model = YOLO('/home/rokey6/C2_py/best.pt')
        self.coordinates = []
        self.output_dir = '/home/rokey6/C2_py/output'
        os.makedirs(self.output_dir, exist_ok=True)

        # Video capture setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to initialize camera. Check the device connection.")
            raise RuntimeError("Camera initialization failed.")
        self.get_logger().info("Camera initialized successfully.")

        self.cap.set(3, 640)
        self.cap.set(4, 480)

        # Control flags
        self.frame_count = 0
        self.lock = threading.Lock()
        self.current_frame = None
        self.processed_frame = None
        self.classNames = ['kid', 'Dummy', 'parent']

        # Start threads
        threading.Thread(target=self.capture_frames, daemon=True).start()
        threading.Thread(target=self.process_frames, daemon=True).start()
        self.timer = self.create_timer(0.1, self.publish_image)
        self.get_logger().info("YOLO Publisher node initialized successfully.")

    def capture_frames(self):
        """Continuously capture frames from the camera in a separate thread."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("Failed to capture frame from camera.")
                continue
            with self.lock:
                self.current_frame = frame.copy()
            time.sleep(0.01)  # Slight delay to control frame rate

    def process_frames(self):
        """Process frames for object detection in a separate thread."""
        while True:
            if self.current_frame is not None and self.frame_count % 2 == 0:
                with self.lock:
                    frame_to_process = self.current_frame.copy()

                # Run YOLO object detection
                try:
                    results = self.model(frame_to_process, stream=True)
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = math.ceil(box.conf[0] * 100) / 100
                            cls = int(box.cls[0])
                            cv2.rectangle(frame_to_process, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame_to_process, f"{self.classNames[cls]}: {confidence}", (x1, y1),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    self.get_logger().info("YOLO detection completed for the frame.")
                except Exception as e:
                    self.get_logger().error(f"Error during YOLO detection: {e}")

                with self.lock:
                    self.processed_frame = frame_to_process

            self.frame_count += 1
            time.sleep(0.05)

    def publish_image(self):
        """Publish the latest processed frame."""
        if self.processed_frame is not None:
            try:
                with self.lock:
                    ros_image = self.bridge.cv2_to_imgmsg(self.processed_frame, encoding="bgr8")
                self.publisher_.publish(ros_image)
                self.get_logger().info("Image published to /processed_image")
            except Exception as e:
                self.get_logger().error(f"Error during image publishing: {e}")

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YoloPublisher()
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f"Exception during node execution: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    main()