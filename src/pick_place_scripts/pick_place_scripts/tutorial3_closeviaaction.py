#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from rclpy.logging import get_logger
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose, PoseStamped
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from control_msgs.action import GripperCommand


# ============================================================
# Scene / Pick-and-Place parameters
# ============================================================

# Cube
CUBE_X = 0.60
CUBE_Y = 0.00
CUBE_Z = 0.03

CUBE_SIZE = 0.05

# Cube top:
#   center = 0.03
#   half size = 0.025
#   top = 0.055
#
# GRASP_HEIGHT is the tool0 target height for the final descent.
# Raised from an earlier 0.06 (only 5mm clearance, caused the goal
# to be in collision) to 0.08 for safer finger clearance above the
# cube's collision geometry.
PRE_GRASP_HEIGHT = 0.43
GRASP_HEIGHT = 0.2

# Floor
#
# Floor center = -0.11
# Floor thickness = 0.02
# Therefore:
#     top = -0.10
#     bottom = -0.12
#
# The floor is below z = 0.
FLOOR_Z = -0.11
FLOOR_SIZE_X = 4.0
FLOOR_SIZE_Y = 4.0
FLOOR_THICKNESS = 0.02

# Gripper orientation for tool0 (pointing straight down)
GRIPPER_ORIENTATION = {
    "x": -0.690,
    "y": 0.724,
    "z": -0.003,
    "w": 0.012,
}

# Gripper open uses the named SRDF pose (set in MoveIt Setup
# Assistant's Robot Poses tab). Gripper CLOSE is done via a direct
# action call, not MoveIt planning - see close_gripper_directly().
GRIPPER_OPEN_CONFIG = "open"
GRIPPER_CLOSE_POSITION = 0.6   # tune this against your actual cube
GRIPPER_CLOSE_MAX_EFFORT = 10.0

# MoveIt planning groups
PLANNING_GROUP_ARM = "ur_manipulator"
PLANNING_GROUP_GRIPPER = "gripper"


# ============================================================
# Helpers
# ============================================================

def make_pose(x, y, z, orientation):
    """Create a PoseStamped in the world frame."""
    pose = PoseStamped()
    pose.header.frame_id = "world"
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    pose.pose.orientation.x = orientation["x"]
    pose.pose.orientation.y = orientation["y"]
    pose.pose.orientation.z = orientation["z"]
    pose.pose.orientation.w = orientation["w"]
    return pose


# ============================================================
# Add collision objects directly to MoveItPy PlanningScene
# ============================================================

def add_collision_objects(planning_scene_monitor, logger):
    """
    Add the floor and pick cube directly to the PlanningScene
    owned by this MoveItPy instance.
    """
    logger.info("Adding collision objects to MoveItPy PlanningScene")

    with planning_scene_monitor.read_write() as scene:

        # ---------------- Floor ----------------
        floor = CollisionObject()
        floor.id = "floor"
        floor.header.frame_id = "world"

        floor_primitive = SolidPrimitive()
        floor_primitive.type = SolidPrimitive.BOX
        floor_primitive.dimensions = [FLOOR_SIZE_X, FLOOR_SIZE_Y, FLOOR_THICKNESS]

        floor_pose = Pose()
        floor_pose.position.x = 0.0
        floor_pose.position.y = 0.0
        floor_pose.position.z = FLOOR_Z
        floor_pose.orientation.w = 1.0

        floor.primitives.append(floor_primitive)
        floor.primitive_poses.append(floor_pose)
        floor.operation = CollisionObject.ADD
        scene.apply_collision_object(floor)

        # ---------------- Pick cube ----------------
        # cube = CollisionObject()
        # cube.id = "pick_cube"
        # cube.header.frame_id = "world"

        # cube_primitive = SolidPrimitive()
        # cube_primitive.type = SolidPrimitive.BOX
        # cube_primitive.dimensions = [CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]

        # cube_pose = Pose()
        # cube_pose.position.x = CUBE_X
        # cube_pose.position.y = CUBE_Y
        # cube_pose.position.z = CUBE_Z
        # cube_pose.orientation.w = 1.0

        # cube.primitives.append(cube_primitive)
        # cube.primitive_poses.append(cube_pose)
        # cube.operation = CollisionObject.ADD
        # scene.apply_collision_object(cube)

        scene.current_state.update()

    logger.info("Collision objects added")


# ============================================================
# Verify planning scene
# ============================================================

def print_planning_scene(planning_scene_monitor, logger):
    logger.info("Checking MoveItPy PlanningScene")

    with planning_scene_monitor.read_only() as scene:
        msg = scene.planning_scene_message

        print("")
        print("==============================================")
        print("       MOVEITPY PLANNING SCENE")
        print("==============================================")
        print("Robot model:", msg.robot_model_name)
        print("Planning frame:", scene.planning_frame)
        print("")
        print("Collision objects:")
        for obj in msg.world.collision_objects:
            print("  ID:", obj.id)
            print("  Frame:", obj.header.frame_id)
            print("  Primitives:", len(obj.primitives))
            for i, primitive in enumerate(obj.primitives):
                print("    primitive", i, "dimensions:", list(primitive.dimensions))
        print("")
        print("Number of collision objects:", len(msg.world.collision_objects))
        print("Attached collision objects:", len(msg.robot_state.attached_collision_objects))
        print("==============================================")
        print("")


# ============================================================
# Plan and execute (OMPL - used for free-space arm motions
# and the named-pose gripper open)
# ============================================================

def plan_and_execute(planning_component, logger, label, robot, group_name, pipeline="ompl", planner_id=None):
    logger.info(f"Planning ({pipeline}): {label}")

    plan_params = PlanRequestParameters(robot)
    plan_params.planning_pipeline = pipeline
    if planner_id is not None:
        plan_params.planner_id = planner_id

    plan_result = planning_component.plan(parameters=plan_params)
    if not plan_result:
        logger.error(f"FAILED to plan: {label} -- aborting sequence")
        return False

    logger.info(f"Executing: {label}")
    robot.execute(group_name, plan_result.trajectory)
    time.sleep(1.0)
    return True


# ============================================================
# Move arm - free-space (OMPL)
# ============================================================

def move_arm_to_pose(arm, logger, x, y, z, label, robot):
    logger.info(f"Setting arm target: x={x:.3f}, y={y:.3f}, z={z:.3f}")
    arm.set_start_state_to_current_state()
    arm.set_goal_state(pose_stamped_msg=make_pose(x, y, z, GRIPPER_ORIENTATION), pose_link="tool0")
    return plan_and_execute(arm, logger, label, robot, PLANNING_GROUP_ARM, pipeline="ompl")


# ============================================================
# Move arm - straight-line descent (Pilz LIN)
# ============================================================

def move_arm_linear_to_pose(arm, logger, x, y, z, label, robot):
    logger.info(f"Setting LINEAR arm target: x={x:.3f}, y={y:.3f}, z={z:.3f}")
    arm.set_start_state_to_current_state()
    arm.set_goal_state(pose_stamped_msg=make_pose(x, y, z, GRIPPER_ORIENTATION), pose_link="tool0")
    return plan_and_execute(
        arm, logger, label, robot, PLANNING_GROUP_ARM,
        pipeline="pilz_industrial_motion_planner", planner_id="LIN"
    )


# ============================================================
# Move gripper - named pose (OMPL, used for "open" only,
# since open never involves colliding with anything)
# ============================================================

def move_gripper_to_named(gripper, logger, configuration_name, label, robot):
    logger.info(f"Setting gripper configuration: {configuration_name}")
    gripper.set_start_state_to_current_state()
    gripper.set_goal_state(configuration_name=configuration_name)
    return plan_and_execute(gripper, logger, label, robot, PLANNING_GROUP_GRIPPER, pipeline="ompl")


# ============================================================
# Close gripper directly via action call (bypasses MoveIt
# collision-checked planning, since closing around the cube is
# SUPPOSED to make contact - MoveIt would never find that "valid")
# ============================================================

def close_gripper_directly(node, logger, position, max_effort=10.0, timeout_sec=5.0):
    client = ActionClient(node, GripperCommand, "/robotiq_gripper_controller/gripper_cmd")

    logger.info("Waiting for gripper action server...")
    if not client.wait_for_server(timeout_sec=timeout_sec):
        logger.error("Gripper action server not available")
        return False

    goal = GripperCommand.Goal()
    goal.command.position = position
    goal.command.max_effort = max_effort

    logger.info(f"Sending direct gripper close command: position={position}")
    send_goal_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_goal_future, timeout_sec=timeout_sec)

    goal_handle = send_goal_future.result()
    if goal_handle is None or not goal_handle.accepted:
        logger.error("Gripper close goal was rejected")
        return False

    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future, timeout_sec=timeout_sec)

    logger.info("Gripper close command completed")
    return True


# ============================================================
# Main
# ============================================================

def main():
    rclpy.init()

    logger = get_logger("pick_test_node")
    logger.info("Starting UR10e + Robotiq MoveItPy pick test")

    moveit_config = (
        MoveItConfigsBuilder("ur10e_robotiq", package_name="ur10e_moveit")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )
    config_dict = moveit_config.to_dict()
    config_dict["planning_pipelines"] = {"pipeline_names": ["ompl", "pilz_industrial_motion_planner"]}

    robot = MoveItPy(node_name="pick_test_node", config_dict=config_dict)
    logger.info("MoveItPy instance created")

    # Let the planning scene monitor receive a few /joint_states
    # messages before we plan anything - avoids a stale "current
    # state" rejection on the very first trajectory.
    time.sleep(2.0)

    # Separate plain rclpy node for the direct gripper action call
    gripper_action_node = Node("gripper_action_client_node")

    planning_scene_monitor = robot.get_planning_scene_monitor()
    add_collision_objects(planning_scene_monitor, logger)
    print_planning_scene(planning_scene_monitor, logger)

    arm = robot.get_planning_component(PLANNING_GROUP_ARM)
    gripper = robot.get_planning_component(PLANNING_GROUP_GRIPPER)
    logger.info("Planning components created")

    # ========================================================
    # PICK SEQUENCE
    # ========================================================

    # 1. Open gripper (named pose, OMPL - fine, nothing to collide with)
    if not move_gripper_to_named(gripper, logger, GRIPPER_OPEN_CONFIG, "Open gripper", robot):
        logger.error("Failed to open gripper")
        gripper_action_node.destroy_node()
        return

    # 2. Move to pre-grasp (free-space, OMPL)
    if not move_arm_to_pose(
        arm, logger, CUBE_X, CUBE_Y, CUBE_Z + PRE_GRASP_HEIGHT, "Move to pre-grasp", robot
    ):
        gripper_action_node.destroy_node()
        return

    # 3. Descend to grasp height (straight line, Pilz LIN)
    if not move_arm_linear_to_pose(
        arm, logger, CUBE_X, CUBE_Y, GRASP_HEIGHT, "Move to grasp (linear)", robot
    ):
        gripper_action_node.destroy_node()
        return

    # 4. Close gripper (direct action call, bypassing MoveIt planning)
    if not close_gripper_directly(
        gripper_action_node, logger, position=GRIPPER_CLOSE_POSITION, max_effort=GRIPPER_CLOSE_MAX_EFFORT
    ):
        logger.error("Failed to close gripper")
        gripper_action_node.destroy_node()
        return

    # 5. Lift (free-space, OMPL)
    if not move_arm_to_pose(
        arm, logger, CUBE_X, CUBE_Y, CUBE_Z + PRE_GRASP_HEIGHT, "Lift cube", robot
    ):
        gripper_action_node.destroy_node()
        return

    logger.info("Pick sequence completed successfully")
    gripper_action_node.destroy_node()


if __name__ == "__main__":
    main()