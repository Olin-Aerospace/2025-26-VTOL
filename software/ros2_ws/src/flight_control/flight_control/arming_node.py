#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from mavros_msgs.srv import CommandBool #ADD DEPENDANCY
#if you import more libraries, add them to dependencies in package.xml

class arm_node(Node):
    def __init__(self):
        super().__init__('armNode')
        self.get_logger().info('arming node has been started')
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        while not self.arming_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /mavros/cmd/arming service...')
        self.req = CommandBool.Request()
    def send_request(self, msg):
        self.req.value = msg#MAKE SURE ITS CALLED VALUE
        self.future = self.arming_client.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

        # Add your node logic here
        # For example, you can create a timer, subscribe to topics, etc.


def main(args=None):
    rclpy.init(args=args)
    node = arm_node()
    response = node.send_request(True)#MIGHT WANT TO MAKE OPPOSITE OF WHAT IT IS not true
    arm_node.get_logger().info("Status of arming:", response.success)#IDK IF ITS SUCCESS
    arm_node.destroy_node()
    #rclpy.spin(node) #Loops the node
    rclpy.shutdown()


if __name__=='__main__':#Optional if you only ever run the node directly with ros2 run 
    main()
