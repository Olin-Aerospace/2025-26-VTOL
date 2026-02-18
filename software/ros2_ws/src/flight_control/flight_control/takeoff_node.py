#!/usr/bin/env python3
#IDK IF THE SHEBANG IS RIGHT
import rclpy
from rclpy.node import Node
from mavros_msgs.srv import CommandTOL
#if you import more libraries, add them to dependencies in package.xml

class takeoff_class(Node):

    def __init__(self):
        super().__init__('takeoff_node')
        self.get_logger().info('takeoff node has been started')
        self.cli = self.create_client(CommandTOL, 'mavros/cmd/takeoff')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = CommandTOL.Request()

    def send_request(self, altitude, lat, lon, min_pitch, yaw):
        self.req.altitude = altitude
        self.req.latitude = lat
        self.req.longitude = lon
        self.req.min_pitch = min_pitch
        self.req.yaw = yaw
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    takeoff_node = takeoff_class()
    altitude, lat, lon, min_pitch, yaw = 2.0,0.0,0.0,0.0,0.0
    response = takeoff_node.send_request(altitude, lat, lon, min_pitch, yaw)
    takeoff_node.get_logger().info(f"results: {response.success} bool and uint f{response.result}")
    takeoff_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()