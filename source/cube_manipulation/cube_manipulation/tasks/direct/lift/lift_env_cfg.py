# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # type: ignore


@configclass
class LiftSceneCfg(InteractiveSceneCfg):
    """场景中所有物体的配置。"""

    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        # spawn=sim_utils.GroundPlaneCfg(),
        spawn=sim_utils.GroundPlaneCfg(
            usd_path="/home/tqp9490/isaac_assets/Assets/Isaac/5.1/Isaac/Environments/Grid/default_environment.usd"
        ),
    )

    # 灯光
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )

    # 机器人底座直接放在地面上（z = 0.0）
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        # 占位符会被替换为 /world/envs/env_0/Robot
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=FRANKA_PANDA_CFG.spawn.replace(
            usd_path="/home/tqp9490/isaac_assets/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd",
            # activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # pos的含义是机器人坐标，分别表示位置（x, y, z）x是前后方向，y是左右方向，z是垂直方向
            # rot的含义是机器人旋转，四元数表示（w, x, y, z）
            # w表示旋转角度 x是roll 左右旋转，y是pitch 前后旋转，z是yaw 水平旋转
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.785,
                "panda_joint3": 0.0,
                "panda_joint4": -2.356,
                "panda_joint5": 0.0,
                "panda_joint6": 1.571,
                "panda_joint7": 0.785,
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
        ),
         actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=0.0,
                damping=40.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=0.0,
                damping=40.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )


    # 方块放在地面上（边长 0.05m，因此中心 z = 0.025m）
    # cube: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.6, 0, 0.025], rot=[1, 0, 0, 0]),
    #     spawn=sim_utils.UsdFileCfg(
    #         # activate_contact_sensors=True,
    #         # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    #         usd_path="/home/tqp9490/isaac_assets/Assets/Isaac/5.1/Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    #         scale=(0.8, 0.8, 0.8),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=1,
    #             max_angular_velocity=1000.0,
    #             max_linear_velocity=1000.0,
    #             max_depenetration_velocity=5.0,
    #             disable_gravity=False,
    #         ),
    #     ),
    # )

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            activate_contact_sensors=True,
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                # 防止穿透
                max_depenetration_velocity=5.0,
                # 重力生效
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
                # 0 表示非金属， 1 表示完全金属
                metallic=0.0,
            ),
            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.025),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


@configclass
class LiftEnvCfg(DirectRLEnvCfg):
    """完整环境配置。"""

    # 仿真
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=2)

    # 场景
    scene: LiftSceneCfg = LiftSceneCfg(
        num_envs=256,
        env_spacing=2.0,
    )

    # RL 基本参数
    decimation: int = 2
    # episode 最长时间
    episode_length_s: float = 7.0

    # 空间维度
    action_space: int = 9               # 7 arm joints + 2 gripper fingers
    observation_space: int = 19         # 见 _get_observations() 注释
    state_space: int = 0

    # cube 大小（边长），用于计算是否成功
    cube_size: float = 0.05
 
    # 动作缩放（将 [-1,1] 的网络输出映射到关节速度）,调整关节速度
    action_scale: float = 0.5