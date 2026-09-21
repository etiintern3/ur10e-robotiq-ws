#!/usr/bin/env python3
"""
Opens the gripper using the predefined 'open' named pose (set earlier in
MoveIt Setup Assistant's Robot Poses tab for the 'gripper' group).

Using the named pose instead of a hardcoded joint value sidesteps any
confusion about which numeric value means "open" vs "closed" - MoveIt
already knows, since you defined it visually in the Setup Assistant.
"""

import rclpy
import time
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit_configs_utils import MoveItConfigsBuilder

PLANNING_GROUP_GRIPPER = "gripper"
GRIPPER_OPEN_CONFIG_NAME = "open"  # must match the name you gave it in Setup Assistant


def plan_and_execute(planning_component, logger, label, robot, group_name):
    logger.info(f"Planning: {label}")
    plan_params = PlanRequestParameters(robot)
    plan_params.planning_pipeline = "ompl"
    plan_result = planning_component.plan(parameters=plan_params)
    if not plan_result:
        logger.error(f"FAILED to plan: {label}")
        return False
    logger.info(f"Executing: {label}")
    robot.execute(group_name, plan_result.trajectory)
    return True


def main():
    rclpy.init()

    moveit_config = (
        MoveItConfigsBuilder("ur10e_robotiq", package_name="ur10e_moveit")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    config_dict = moveit_config.to_dict()
    config_dict["planning_pipelines"] = {"pipeline_names": ["ompl"]}

    robot = MoveItPy(node_name="open_gripper_node", config_dict=config_dict)
    logger = rclpy.logging.get_logger("open_gripper_node")

    gripper = robot.get_planning_component(PLANNING_GROUP_GRIPPER)
    time.sleep(2.0) 

    # Set start state to wherever the robot currently is (e.g. spawned closed)
    gripper.set_start_state_to_current_state()

    # Goal state: the predefined "open" configuration from Setup Assistant
    gripper.set_goal_state(configuration_name=GRIPPER_OPEN_CONFIG_NAME)

    if plan_and_execute(gripper, logger, "open gripper (named pose)", robot, PLANNING_GROUP_GRIPPER):
        logger.info("Gripper should now be open. Check RViz.")
    else:
        logger.error(
            "Planning to the 'open' named pose failed. Double-check the name "
            "matches exactly what you saved in Setup Assistant's Robot Poses tab."
        )

    rclpy.shutdown()


if __name__ == "__main__":
    main()