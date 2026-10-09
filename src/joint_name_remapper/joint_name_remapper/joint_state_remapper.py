#!/usr/bin/env python3
"""
joint_state_remapper.py

Subscribes to a JointState topic using MoveIt/ros2_control joint names
(e.g. the Robotiq gripper's "robotiq_85_*" names), renames the entries
to match Isaac Sim's USD joint names, and republishes on a new topic.

Point Isaac Sim's "ROS2 Subscribe Joint State" node's Topic Name field
at the OUTPUT topic instead of the original one.

Usage:
    python3 joint_state_remapper.py --ros-args \
        -p input_topic:=/joint_command \
        -p output_topic:=/joint_command_isaac
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


# MoveIt / URDF name  ->  Isaac Sim USD joint name
# Edit this to match your setup exactly.
NAME_MAP = {
    "robotiq_85_left_knuckle_joint": "finger_joint",
    "robotiq_85_right_knuckle_joint": "right_outer_knuckle_joint",
    "robotiq_85_right_finger_tip_joint": "right_inner_finger_joint",
    "robotiq_85_right_inner_knuckle_joint": "right_inner_finger_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint": "left_inner_finger_knuckle_joint",
    "robotiq_85_left_finger_tip_joint": "left_inner_finger_joint",
}


class JointStateRemapper(Node):
    def __init__(self):
        super().__init__("joint_state_remapper")

        # Adjust these defaults to match your actual topic names,
        # or override them at launch with --ros-args -p input_topic:=... etc.
        self.declare_parameter("input_topic", "/joint_states")
        self.declare_parameter("output_topic", "/joint_command_isaac")

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value

        self.pub = self.create_publisher(JointState, output_topic, 10)
        self.sub = self.create_subscription(
            JointState, input_topic, self.callback, 10
        )

        unmapped_warned = set()
        self._unmapped_warned = unmapped_warned

        self.get_logger().info(
            f"Remapping joint names: '{input_topic}' -> '{output_topic}'"
        )
        self.get_logger().info(f"Loaded {len(NAME_MAP)} name mappings.")

    def callback(self, msg: JointState):
        remapped = JointState()
        remapped.header = msg.header

        new_names = []
        for n in msg.name:
            if n in NAME_MAP:
                new_names.append(NAME_MAP[n])
            else:
                new_names.append(n)  # pass through unchanged (e.g. arm joints)
                if n not in self._unmapped_warned:
                    self.get_logger().debug(f"No mapping for '{n}', passing through.")
                    self._unmapped_warned.add(n)

        remapped.name = new_names
        remapped.position = msg.position
        remapped.velocity = msg.velocity
        remapped.effort = msg.effort

        self.pub.publish(remapped)


def main(args=None):
    rclpy.init(args=args)
    node = JointStateRemapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()