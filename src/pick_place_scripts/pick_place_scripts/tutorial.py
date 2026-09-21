#!/usr/bin/env python3

import time

import rclpy
from rclpy.logging import get_logger

from geometry_msgs.msg import Pose, PoseStamped
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive


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
# The tool target at GRASP_HEIGHT = 0.06
# is therefore 5 mm above the cube.

PRE_GRASP_HEIGHT = 0.23
GRASP_HEIGHT = 0.06

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

# Gripper orientation for tool0
GRIPPER_ORIENTATION = {
    "x": -0.690,
    "y": 0.724,
    "z": -0.003,
    "w": 0.012,
}

GRIPPER_OPEN_POSITION = 0.1
GRIPPER_CLOSED_POSITION = 0.79

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

    This follows the same approach as the official MoveItPy
    planning-scene tutorial:
        read_write() -> apply_collision_object()
    """

    logger.info("Adding collision objects to MoveItPy PlanningScene")

    with planning_scene_monitor.read_write() as scene:

        # --------------------------------------------------------
        # Floor
        # --------------------------------------------------------

        floor = CollisionObject()

        floor.id = "floor"
        floor.header.frame_id = "world"

        floor_primitive = SolidPrimitive()
        floor_primitive.type = SolidPrimitive.BOX
        floor_primitive.dimensions = [
            FLOOR_SIZE_X,
            FLOOR_SIZE_Y,
            FLOOR_THICKNESS,
        ]

        floor_pose = Pose()

        floor_pose.position.x = 0.0
        floor_pose.position.y = 0.0
        floor_pose.position.z = FLOOR_Z

        floor_pose.orientation.x = 0.0
        floor_pose.orientation.y = 0.0
        floor_pose.orientation.z = 0.0
        floor_pose.orientation.w = 1.0

        floor.primitives.append(floor_primitive)
        floor.primitive_poses.append(floor_pose)

        floor.operation = CollisionObject.ADD

        scene.apply_collision_object(floor)

        # --------------------------------------------------------
        # Pick cube
        # --------------------------------------------------------

        cube = CollisionObject()

        cube.id = "pick_cube"
        cube.header.frame_id = "world"

        cube_primitive = SolidPrimitive()
        cube_primitive.type = SolidPrimitive.BOX
        cube_primitive.dimensions = [
            CUBE_SIZE,
            CUBE_SIZE,
            CUBE_SIZE,
        ]

        cube_pose = Pose()

        cube_pose.position.x = CUBE_X
        cube_pose.position.y = CUBE_Y
        cube_pose.position.z = CUBE_Z

        cube_pose.orientation.x = 0.0
        cube_pose.orientation.y = 0.0
        cube_pose.orientation.z = 0.0
        cube_pose.orientation.w = 1.0

        cube.primitives.append(cube_primitive)
        cube.primitive_poses.append(cube_pose)

        cube.operation = CollisionObject.ADD

        scene.apply_collision_object(cube)

        # --------------------------------------------------------
        # Make sure the planning scene state is updated
        # --------------------------------------------------------

        scene.current_state.update()

    logger.info("Collision objects added")


# ============================================================
# Verify planning scene
# ============================================================

def print_planning_scene(planning_scene_monitor, logger):
    """
    Print the collision objects currently known to the
    MoveItPy PlanningScene.
    """

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

                print(
                    "    primitive",
                    i,
                    "dimensions:",
                    list(primitive.dimensions),
                )

        print("")
        print(
            "Number of collision objects:",
            len(msg.world.collision_objects),
        )

        print(
            "Attached collision objects:",
            len(msg.robot_state.attached_collision_objects),
        )

        print("==============================================")
        print("")


# ============================================================
# Plan and execute
# ============================================================

def plan_and_execute(
    planning_component,
    logger,
    label,
    robot,
    group_name,
):
    """
    Plan and execute a trajectory.
    """

    logger.info(f"Planning: {label}")

    plan_params = PlanRequestParameters(robot)

    plan_params.planning_pipeline = "ompl"

    plan_result = planning_component.plan(
        parameters=plan_params
    )

    if not plan_result:

        logger.error(
            f"FAILED to plan: {label} -- aborting sequence"
        )

        return False

    logger.info(f"Executing: {label}")

    robot_trajectory = plan_result.trajectory

    robot.execute(
        group_name,
        robot_trajectory,
    )

    time.sleep(1.0)

    return True


# ============================================================
# Move arm to pose
# ============================================================

def move_arm_to_pose(
    arm,
    logger,
    x,
    y,
    z,
    label,
    robot,
):
    """
    Plan the UR10e tool0 to a Cartesian pose.
    """

    logger.info(
        f"Setting arm target: "
        f"x={x:.3f}, y={y:.3f}, z={z:.3f}"
    )

    arm.set_start_state_to_current_state()

    arm.set_goal_state(
        pose_stamped_msg=make_pose(
            x,
            y,
            z,
            GRIPPER_ORIENTATION,
        ),
        pose_link="tool0",
    )

    return plan_and_execute(
        arm,
        logger,
        label,
        robot,
        PLANNING_GROUP_ARM,
    )


# ============================================================
# Move gripper
# ============================================================

# def move_gripper_to(
#     gripper,
#     logger,
#     joint_value,
#     label,
#     robot,
# ):
#     """
#     Plan and execute a Robotiq gripper movement.
#     """

#     logger.info(
#         f"Setting gripper position: {joint_value}"
#     )

#     gripper.set_start_state_to_current_state()

#     robot_state = gripper.get_start_state()

#     robot_state.set_joint_group_positions(
#         PLANNING_GROUP_GRIPPER,
#         [joint_value],
#     )

#     gripper.set_goal_state(
#         robot_state=robot_state
#     )

#     return plan_and_execute(
#         gripper,
#         logger,
#         label,
#         robot,
#         PLANNING_GROUP_GRIPPER,
#     )

def move_gripper_to(
    gripper,
    logger,
    configuration_name,
    label,
    robot,
):
    logger.info(f"Setting gripper configuration: {configuration_name}")

    # Start from the robot's actual current state
    gripper.set_start_state_to_current_state()

    # Use the named configuration from the SRDF
    gripper.set_goal_state(
        configuration_name=configuration_name
    )

    return plan_and_execute(
        gripper,
        logger,
        label,
        robot,
        PLANNING_GROUP_GRIPPER,
    )

# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # ROS initialization
    # --------------------------------------------------------

    rclpy.init()

    logger = get_logger("pick_test_node")

    logger.info("Starting UR10e + Robotiq MoveItPy pick test")

    # --------------------------------------------------------
    # MoveIt configuration
    # --------------------------------------------------------

    moveit_config = (
        MoveItConfigsBuilder(
            "ur10e_robotiq",
            package_name="ur10e_moveit",
        )
        .planning_pipelines(
            pipelines=["ompl"]
        )
        .to_moveit_configs()
    )

    config_dict = moveit_config.to_dict()

    config_dict["planning_pipelines"] = {
        "pipeline_names": ["ompl"]
    }

    # --------------------------------------------------------
    # Create MoveItPy
    # --------------------------------------------------------

    robot = MoveItPy(
        node_name="pick_test_node",
        config_dict=config_dict,
    )

    logger.info("MoveItPy instance created")

    # --------------------------------------------------------
    # Get PlanningSceneMonitor
    # --------------------------------------------------------

    planning_scene_monitor = (
        robot.get_planning_scene_monitor()
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Add floor + cube directly to THIS MoveItPy scene.
    # --------------------------------------------------------

    add_collision_objects(
        planning_scene_monitor,
        logger,
    )

    # --------------------------------------------------------
    # Verify scene before planning
    # --------------------------------------------------------

    print_planning_scene(
        planning_scene_monitor,
        logger,
    )

    # --------------------------------------------------------
    # Get planning components
    # --------------------------------------------------------

    arm = robot.get_planning_component(
        PLANNING_GROUP_ARM
    )

    gripper = robot.get_planning_component(
        PLANNING_GROUP_GRIPPER
    )

    logger.info("Planning components created")

    # ========================================================
    # PICK AND PLACE SEQUENCE
    # ========================================================

    
    # --------------------------------------------------------
    # 1. Open gripper
    # --------------------------------------------------------
    if not move_gripper_to(
        gripper,
        logger,
        "open",
        "Open gripper",
        robot,
    ):
        logger.error("Failed to open gripper")
        return
    # if not move_gripper_to(
    #     gripper,
    #     logger,
    #     GRIPPER_OPEN_POSITION,
    #     "Open gripper",
    #     robot,
    # ):
    #     return

    # --------------------------------------------------------
    # 2. Move to pre-grasp
    # --------------------------------------------------------

    # if not move_arm_to_pose(
    #     arm,
    #     logger,
    #     CUBE_X,
    #     CUBE_Y,
    #     CUBE_Z + PRE_GRASP_HEIGHT,
    #     "Move to pre-grasp",
    #     robot,
    # ):
    #     return

    # # --------------------------------------------------------
    # # 3. Move down to grasp
    # # --------------------------------------------------------

    # if not move_arm_to_pose(
    #     arm,
    #     logger,
    #     CUBE_X,
    #     CUBE_Y,
    #     GRASP_HEIGHT,
    #     "Move to grasp",
    #     robot,
    # ):
    #     return

    # # --------------------------------------------------------
    # # 4. Close gripper
    # # --------------------------------------------------------

    # if not move_gripper_to(
    #     gripper,
    #     logger,
    #     GRIPPER_CLOSED_POSITION,
    #     "Close gripper",
    #     robot,
    # ):
    #     return

    # # --------------------------------------------------------
    # # 5. Lift cube
    # # --------------------------------------------------------

    # if not move_arm_to_pose(
    #     arm,
    #     logger,
    #     CUBE_X,
    #     CUBE_Y,
    #     CUBE_Z + PRE_GRASP_HEIGHT,
    #     "Lift cube",
    #     robot,
    # ):
    #     return

    logger.info("Pick sequence completed successfully")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()