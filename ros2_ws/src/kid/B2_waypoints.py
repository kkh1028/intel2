import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import FollowWaypoints
import math
import threading
import sys
import select
import termios
import tty

class WaypointFollower(Node):
    def __init__(self):
        super().__init__('waypoint_follower')
        self.action_client = ActionClient(self, FollowWaypoints, '/follow_waypoints')
        self.direction = 1  # 1 for forward, -1 for backward

    def euler_to_quaternion(self, roll, pitch, yaw):
        # Convert Euler angles to a quaternion
        qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
        qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
        qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
        qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
        return Quaternion(x=qx, y=qy, z=qz, w=qw)

    def send_goal(self):
        # 웨이포인트 정의
        waypoints = []

        # 초기 위치
        initial_pose = PoseStamped()
        initial_pose.header.stamp.sec = 0
        initial_pose.header.stamp.nanosec = 0
        initial_pose.header.frame_id = "map"
        initial_pose.pose.position.x = 0.11807980388402939
        initial_pose.pose.position.y = 0.17090362310409546
        initial_pose.pose.position.z = 0.0
        initial_pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=-1.8764856488624068e-06, w=0.9999999999982394)
        waypoints.append(initial_pose)

        # 웨이포인트 1
        waypoint1 = PoseStamped()
        waypoint1.header.stamp.sec = 0
        waypoint1.header.stamp.nanosec = 0
        waypoint1.header.frame_id = "map"
        waypoint1.pose.position.x = 0.3569794298908797
        waypoint1.pose.position.y = -0.6187271844639903
        waypoint1.pose.position.z = 0.0
        waypoint1.pose.orientation = Quaternion(x=0.0, y=0.0, z=-0.6813896186371246, w=0.7319208889036806)
        waypoints.append(waypoint1)

        # 웨이포인트 2
        waypoint2 = PoseStamped()
        waypoint2.header.stamp.sec = 0
        waypoint2.header.stamp.nanosec = 0
        waypoint2.header.frame_id = "map"
        waypoint2.pose.position.x = -0.743714304133461
        waypoint2.pose.position.y = -0.7280323908901938
        waypoint2.pose.position.z = 0.0
        waypoint2.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.6893326348670691, w=0.7244449727254817)
        waypoints.append(waypoint2)

        # 웨이포인트 3
        waypoint3 = PoseStamped()
        waypoint3.header.stamp.sec = 0
        waypoint3.header.stamp.nanosec = 0
        waypoint3.header.frame_id = "map"
        waypoint3.pose.position.x = -0.8159513693570857
        waypoint3.pose.position.y = -0.09763397220546177
        waypoint3.pose.position.z = 0.0
        waypoint3.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.9969340447193733, w=0.078246472632769)
        waypoints.append(waypoint3)

        # 웨이포인트 4
        waypoint3 = PoseStamped()
        waypoint3.header.stamp.sec = 0
        waypoint3.header.stamp.nanosec = 0
        waypoint3.header.frame_id = "map"
        waypoint3.pose.position.x = -1.369369940041623
        waypoint3.pose.position.y = -0.01831099600081316
        waypoint3.pose.position.z = 0.0
        waypoint3.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.9986154892615471, w=0.05260327562919393)
        waypoints.append(waypoint3)

        # 웨이포인트 5
        waypoint4 = PoseStamped()
        waypoint4.header.stamp.sec = 0
        waypoint4.header.stamp.nanosec = 0
        waypoint4.header.frame_id = "map"
        waypoint4.pose.position.x = -1.401132613168534
        waypoint4.pose.position.y = -0.6474258592152898
        waypoint4.pose.position.z = 0.0
        waypoint4.pose.orientation = Quaternion(x=0.0, y=0.0, z=-0.9677908519970032, w=0.2517555695330585)
        waypoints.append(waypoint4)

        if self.direction == -1:
            waypoints.reverse()

        # FollowWaypoints 액션 목표 생성 및 전송
        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = waypoints

        # 서버 연결 대기
        self.action_client.wait_for_server()

        # 목표 전송 및 피드백 콜백 설정
        self._send_goal_future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Current Waypoint Index: {feedback.current_waypoint}')

    def cancel_goal(self):
        if self._goal_handle is not None:
            self.get_logger().info('Attempting to cancel the goal...')
            cancel_future = self._goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_done_callback)
        else:
            self.get_logger().info('No active goal to cancel.')

    def cancel_done_callback(self, future):
        cancel_response = future.result()
        if len(cancel_response.goals_cancelled) > 0:
            self.get_logger().info('Goal cancellation accepted. Exiting program...')
            self.destroy_node()
            rclpy.shutdown()
            sys.exit(0)  # Exit the program after successful cancellation
        else:
            self.get_logger().info('Goal cancellation failed or no active goal to cancel.')

    def get_result_callback(self, future):
        result = future.result().result
        missed_waypoints = result.missed_waypoints
        if missed_waypoints:
            self.get_logger().info(f'Missed waypoints: {missed_waypoints}')
        else:
            self.get_logger().info('All waypoints completed successfully!')

        # 방향 전환 후 목표 다시 전송
        self.direction *= -1
        self.get_logger().info('Switching direction and resending goal...')
        self.send_goal()

def keyboard_listener(node):
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                if key.lower() == 'g':
                    node.get_logger().info('Key "g" pressed. Sending goal...')
                    node.send_goal()
                elif key.lower() == 's':
                    node.get_logger().info('Key "s" pressed. Cancelling goal...')
                    node.cancel_goal()
                    break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollower()
    
    thread = threading.Thread(target=keyboard_listener, args=(node,), daemon=True)
    thread.start()
    
    rclpy.spin(node)

if __name__ == '__main__':
    main()
