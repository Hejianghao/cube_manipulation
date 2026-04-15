# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv
# from isaaclab.sensors import ContactSensor

from .lift_env_cfg import LiftEnvCfg

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

def define_hand_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/handMarkers",
        markers={
            "hand_axis": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 1.0)
                ),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)

def define_grasp_center_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/grasp_center",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.005,   
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),   # 绿色
                ),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)

class LiftEnv(DirectRLEnv):
    cfg: LiftEnvCfg

    def __init__(self, cfg: LiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 关节索引（Franka: 0-6 是手臂, 7-8 是夹爪）
        self._arm_joint_ids = list(range(7))
        self._gripper_joint_ids = [7, 8]
        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._processed_actions = torch.zeros_like(self._actions)

        self._lift_height_threshold = 0.03  # cube 需上升的高度 (m)，用于判断 grasp 阶段是否成功
        self._grasp_distance_threshold = 0.02
        self._cube_rest_height = 0.025

        self.episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._completed_episode_count = 0
        self._successful_episode_count = 0
        self._cumulative_success_rate = 0.0
        self._panda_hand_idx: int | None = None
        self._left_finger_idx: int | None = None
        self._right_finger_idx: int | None = None
        self._reach_metrics_cache: dict[str, torch.Tensor] | None = None
        self._tip_offset_local = torch.tensor([[0.0, 0.0, 0.05]], device=self.device)
        self._basis_x = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        self._basis_y = torch.tensor([[0.0, 1.0, 0.0]], device=self.device)
        self._basis_z = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)

        # 可视化 marker：显示 grasp_center 位置，帮助调试和理解 agent 行为
        self.debug_vis = False
        if self.debug_vis:
            # to show marker
            self.hand_marker = define_hand_markers()
            self.grasp_center_marker = define_grasp_center_markers()
            
       
    # ------------------------------------------------------------------
    # 场景搭建
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.robot = self.scene.articulations["robot"]
        self.cube = self.scene.rigid_objects["cube"]
        # copy_from_source=False 避免重复加载资源，提升性能
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.scene.ground.prim_path])
    

    def _visualize_hand_marker(self):
        ee_idx = self._get_panda_hand_idx()

        # panda_hand 世界位置与姿态
        hand_pos = self.robot.data.body_pos_w[:, ee_idx, :].clone()      # (N, 3)
        hand_quat = self.robot.data.body_quat_w[:, ee_idx, :].clone()    # (N, 4)

        # 验证 panda_hand 的局部坐标轴方向，确保 marker 旋转正确
        # x_w = self._quat_rotate(hand_quat, torch.tensor([[1.0, 0.0, 0.0]], device=self.device).repeat(self.num_envs, 1))
        # y_w = self._quat_rotate(hand_quat, torch.tensor([[0.0, 1.0, 0.0]], device=self.device).repeat(self.num_envs, 1))
        # z_w = self._quat_rotate(hand_quat, torch.tensor([[0.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1))

        # print("x_w:", x_w[0])
        # print("y_w:", y_w[0])
        # print("z_w:", z_w[0])

        # 稍微往上抬一点，避免和模型重合看不清
        hand_pos[:, 0] += 0.2

        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
        # 旋转 marker, 使它指向 -z 方向。确认 z 方向才是panda_hand 的局部向下方向
        axis_offset = torch.tensor([0.70710678, 0.0, -0.70710678, 0.0], device=hand_quat.device, dtype=hand_quat.dtype)
        axis_offset = axis_offset.unsqueeze(0).repeat(hand_quat.shape[0], 1)
        marker_quat = self._quat_multiply(hand_quat, axis_offset)
        self.hand_marker.visualize(hand_pos, marker_quat, marker_indices=indices)

    def _visualize_grasp_center(self):
        grasp_center = self._get_grasp_center()
        marker_orientations = torch.zeros((grasp_center.shape[0], 4), device=grasp_center.device)
        marker_orientations[:, 0] = 1.0
        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
        self.grasp_center_marker.visualize(grasp_center, marker_orientations, marker_indices=indices)
 
    # ------------------------------------------------------------------
    # 动作处理
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._reach_metrics_cache = None
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions

        if self.debug_vis:
            self._visualize_hand_marker()
            self._visualize_grasp_center()

    def _apply_action(self) -> None:
        # 将 position target 设为当前位置，消除 k_p 项的干扰，
        # 使力矩完全由 velocity target（k_d 项）决定
        self.robot.set_joint_position_target(self.robot.data.joint_pos.clone())
        self.robot.set_joint_velocity_target(self._processed_actions)

    # ------------------------------------------------------------------
    # 辅助：坐标与四元数
    # ------------------------------------------------------------------

    def _get_body_indices(self) -> tuple[int, int, int]:
        if self._panda_hand_idx is None:
            body_names = self.robot.data.body_names
            self._panda_hand_idx = body_names.index("panda_hand")
            self._left_finger_idx = body_names.index("panda_leftfinger")
            self._right_finger_idx = body_names.index("panda_rightfinger")

        assert self._panda_hand_idx is not None
        assert self._left_finger_idx is not None
        assert self._right_finger_idx is not None
        return self._panda_hand_idx, self._left_finger_idx, self._right_finger_idx

    def _get_panda_hand_idx(self) -> int:
        panda_hand_idx, _, _ = self._get_body_indices()
        return panda_hand_idx

    # ee 的世界坐标
    def _get_ee_pos(self) -> torch.Tensor:
        """返回 panda_hand 在世界坐标系下的位置。"""
        ee_idx = self._get_panda_hand_idx()
        ee_pos_world = self.robot.data.body_pos_w[:, ee_idx, :]
        return ee_pos_world
    
    # fingertip 中点的 世界坐标
    def _get_grasp_center(self) -> torch.Tensor:
        """返回左右手指根部中点的世界坐标，作为抓取参考点。

        比用 panda_hand 原点更接近真实抓取位置。
        注意：body_pos_w 返回的是 link 原点，Franka 手指原点在指根而非指尖。
        """
        # 找到左右手指 link 的 index
        _, left_idx, right_idx = self._get_body_indices()

        # 手指 link 原点的世界坐标
        left_pos = self.robot.data.body_pos_w[:, left_idx, :]      # (N, 3)
        right_pos = self.robot.data.body_pos_w[:, right_idx, :]    # (N, 3)

        # 手指 link 的世界朝向四元数
        left_quat = self.robot.data.body_quat_w[:, left_idx, :]    # (N, 4)
        right_quat = self.robot.data.body_quat_w[:, right_idx, :]  # (N, 4)
        
        # 变成 N 个 local 的 3D 向量，表示从手指根部到指尖的偏移
        # 已经验证过，Franka 手指 link 的局部 z 轴确实是指向指尖的
        tip_offset_local = self._tip_offset_local.to(device=left_pos.device, dtype=left_pos.dtype).expand(left_pos.shape[0], -1)

        # link的原点坐标，加上 偏移了夹子长度的世界坐标，就是指尖坐标
        left_tip = left_pos + self._quat_rotate(left_quat, tip_offset_local)     # (N, 3)
        right_tip = right_pos + self._quat_rotate(right_quat, tip_offset_local)  # (N, 3)

        # 左右指尖中点
        fingertip_center = (left_tip + right_tip) / 2.0   # (N, 3)

        return fingertip_center                    # (N, 3)

    def _quat_conjugate(self, q: torch.Tensor) -> torch.Tensor:
        """四元数共轭，即反向旋转。输入格式 (w, x, y, z)。"""
        q_conj = q.clone()
        q_conj[:, 1:] = -q_conj[:, 1:]
        return q_conj

    def _quat_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """四元数乘法。输入输出格式均为 (w, x, y, z)。"""
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dim=-1)

    def _quat_rotate(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """用四元数 q 旋转三维向量 v（三明治公式 q·v·q*）。"""
        zeros = torch.zeros((v.shape[0], 1), device=v.device, dtype=v.dtype)
        v_quat = torch.cat([zeros, v], dim=-1)
        return self._quat_multiply(
            self._quat_multiply(q, v_quat),
            self._quat_conjugate(q),
        )[:, 1:]

    def _quat_rotate_inverse(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """将世界坐标系下的向量 v 转换到 q 对应的局部坐标系。"""
        return self._quat_rotate(self._quat_conjugate(q), v)

    def _get_robot_root_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 robot root 在世界坐标系下的位置与朝向。"""
        return self.robot.data.root_pos_w, self.robot.data.root_quat_w

    def _world_to_robot_root_frame(self, pos_w: torch.Tensor) -> torch.Tensor:
        """将世界坐标系下的位置转换到 robot root 坐标系。"""
        root_pos_w, root_quat_w = self._get_robot_root_pose()
        return self._quat_rotate_inverse(root_quat_w, pos_w - root_pos_w)

    def _quat_to_robot_root_frame(self, quat_w: torch.Tensor) -> torch.Tensor:
        """将世界坐标系下的姿态转换到 robot root 坐标系。"""
        _, root_quat_w = self._get_robot_root_pose()
        return self._quat_multiply(self._quat_conjugate(root_quat_w), quat_w)

    # ------------------------------------------------------------------
    # 核心几何量
    # ------------------------------------------------------------------

    def _normalize(self, vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """将向量归一化，避免长度过小时出现数值不稳定。"""
        return vec / torch.clamp(torch.norm(vec, dim=-1, keepdim=True), min=eps)

    def _get_cube_axes(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 cube 局部坐标轴在世界坐标系下的方向。

        虽然当前 reset 时 cube 不旋转，但这里仍然从四元数计算，
        这样以后若加入 cube yaw 随机化，这套对齐定义可以直接复用。
        """
        cube_quat = self.cube.data.root_quat_w  # (N, 4)

        basis_x = self._basis_x.to(device=cube_quat.device, dtype=cube_quat.dtype).expand(self.num_envs, -1)
        basis_y = self._basis_y.to(device=cube_quat.device, dtype=cube_quat.dtype).expand(self.num_envs, -1)
        basis_z = self._basis_z.to(device=cube_quat.device, dtype=cube_quat.dtype).expand(self.num_envs, -1)

        cube_x_axis = self._normalize(self._quat_rotate(cube_quat, basis_x))
        cube_y_axis = self._normalize(self._quat_rotate(cube_quat, basis_y))
        cube_z_axis = self._normalize(self._quat_rotate(cube_quat, basis_z))
        return cube_x_axis, cube_y_axis, cube_z_axis

    def _get_gripper_closing_axis(self) -> torch.Tensor:
        """返回夹爪闭合方向在世界坐标系下的单位向量。

        这里用左右手指根部连线来近似夹爪的闭合方向。该方向与 cube 顶面边缘对齐时，
        夹爪在平面内的 yaw 姿态更适合抓取方块。
        """
        _, left_idx, right_idx = self._get_body_indices()

        left_pos = self.robot.data.body_pos_w[:, left_idx, :]
        right_pos = self.robot.data.body_pos_w[:, right_idx, :]
        closing_axis = right_pos - left_pos
        return self._normalize(closing_axis)

    def _get_grasp_alignment_metrics(self) -> dict[str, torch.Tensor]:
        """计算“夹子与 cube 对齐”的几何指标。

        对齐拆成两部分：
        1. approach_alignment：夹爪接近轴是否更接近 top-down 姿态（软偏好）
        2. closing_alignment：夹爪闭合方向是否与 cube 顶面的某一条边平行

        由于方块顶面是正方形，夹爪在平面内与 cube 的 x 边或 y 边对齐都算正确，
        因此 closing_alignment 取这两个方向中的较大值。
        """
        ee_idx = self._get_panda_hand_idx()
        ee_quat = self.robot.data.body_quat_w[:, ee_idx, :]  # (N, 4)

        # panda_hand 的局部 +z 轴是夹爪的接近轴，变换到世界坐标系后用于判断 top-down 姿态
        local_approach_axis = self._basis_z.to(device=ee_quat.device, dtype=ee_quat.dtype).expand(self.num_envs, -1)
        ee_approach_axis_w = self._normalize(self._quat_rotate(ee_quat, local_approach_axis))

        gripper_closing_axis = self._get_gripper_closing_axis()
        world_up = self._basis_z.to(device=ee_quat.device, dtype=ee_quat.dtype).expand(self.num_envs, -1)

        # 接近轴与 cube 顶面法向反方向越接近，越像从上往下抓。
        # 只奖励 top-down 半球：0 度时为 1，90 度时降到 0，超过 90 度后保持 0，不再继续惩罚。
        approach_alignment = torch.clamp(torch.sum(ee_approach_axis_w * (-world_up), dim=-1), 0.0, 1.0)
        
        z_component = torch.abs(torch.sum(gripper_closing_axis * world_up, dim=-1))
        horizontal_alignment = 1.0 - z_component


        alignment_score = 0.3 * approach_alignment + 0.7 * horizontal_alignment

        return {
            "approach_alignment": approach_alignment,
            "horizontal_alignment": horizontal_alignment,
            "alignment_score": alignment_score,
        }
    
    # def _get_lateral_offset(self) -> torch.Tensor:
    #     """计算 grasp_center 相对于 cube 的 lateral offset，即在夹爪闭合轴方向上的偏移量。
    #     几何定义：
    #         lateral_offset：cube 中心相对 grasp_center 在闭合轴方向的偏移（绝对值）。
    #     """

    #     cube_pos     = self.cube.data.root_pos_w                   # (N, 3)
    #     grasp_center = self._get_grasp_center()    

    #     rel = cube_pos - grasp_center                                        # (N, 3)
    #     closing_axis   = self._get_gripper_closing_axis()                    # (N, 3)
    #     lateral_offset = torch.abs(torch.sum(rel * closing_axis, dim=-1))   # (N,)
    #     # clamp 0 - 1
    #     lateral_offset = torch.clamp(lateral_offset, min=0.0, max=1.0)
    #     return lateral_offset

    def _get_ee_velocity_metric(self) -> torch.Tensor:
        ee_idx = self.robot.body_names.index("panda_hand")
        ee_lin_vel = self.robot.data.body_lin_vel_w[:, ee_idx, :]   # (N, 3)
        ee_speed = torch.norm(ee_lin_vel, dim=-1)                        # (N,) m/s
        return ee_speed

    def _get_reach_metrics(self) -> dict[str, torch.Tensor]:
        """计算 reach 阶段所需的几何量。

        返回：
            grasp_center : (N, 3) 夹爪抓取中心，世界坐标
            cube_pos     : (N, 3) cube 中心，世界坐标
            distance     : (N,)   grasp_center 到 cube 的欧氏距离
            success      : (N,)   reach 成功条件：距离够近 且 左右对准
        """
        if self._reach_metrics_cache is not None:
            return self._reach_metrics_cache

        cube_pos     = self.cube.data.root_pos_w                   # (N, 3)
        grasp_center = self._get_grasp_center()                    # (N, 3)

        # cube 相对于 grasp_center 的向量（世界坐标系）
        rel_world = cube_pos - grasp_center                        # (N, 3)

        #0 欧氏距离（不分方向）
        distance = torch.norm(rel_world, dim=-1)                   # (N,)

        #1 gripper 平均开度：两个手指关节角度的平均值，越大越开
        gripper_pos = self.robot.data.joint_pos[:, self._gripper_joint_ids]   # (N, 2)
        gripper_mean = gripper_pos.mean(dim=-1)   # (N,)

        #2 末端执行器速度（世界坐标系下的线速度大小）
        ee_velocity = self._get_ee_velocity_metric()

        #3 cube 当前中心高度相对初始静止高度的抬升量。先减去世界坐标原点的高度，再减去 cube 的 resting height，得到相对于地面的高度。
        cube_height_local = cube_pos[:, 2] - self.scene.env_origins[:, 2]
        lift_height = cube_height_local - self._cube_rest_height

        #4 平行关系
        alignment = self._get_grasp_alignment_metrics()
        alignment_score = alignment["alignment_score"]

        # 几何抓取判定：gripper 合拢到一定程度 + 指尖中心离 cube 足够近
        geo_grasped = (gripper_mean <= 0.025) & (distance < 0.02) & (lift_height > 0.005)

        is_grasped = geo_grasped

        success = torch.logical_and(is_grasped, lift_height >= self._lift_height_threshold)

        metrics = {
            "grasp_center": grasp_center,
            "cube_pos":     cube_pos,
            "distance":     distance,
            "alignment_score": alignment_score,
            "gripper_mean": gripper_mean,
            "ee_velocity": ee_velocity,
            "lift_height": lift_height,
            "is_grasped": is_grasped,
            "geo_grasped": geo_grasped,
            "success":      success,
        }
        self._reach_metrics_cache = metrics
        return metrics

    # ------------------------------------------------------------------
    # Observation / Reward / Done
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        """构建 22 维观测向量：

          [0:9]   关节角度（手臂 7 + 夹爪 2）
          [9:12]  末端执行器位置（robot root 坐标）
          [12:16] 末端执行器朝向四元数（robot root 坐标）
          [16:19] cube 位置（robot root 坐标）
          [19:22] cube 相对 grasp_center 的位移向量（robot root 坐标）
        """
        joint_pos = self.robot.data.joint_pos                                 # (N, 9)

        ee_idx = self._get_panda_hand_idx()
        ee_quat_w = self.robot.data.body_quat_w[:, ee_idx, :]                 # (N, 4)

        metrics = self._get_reach_metrics()
        cube_pos_w = metrics["cube_pos"]                                      # (N, 3)
        grasp_center_w = metrics["grasp_center"]                              # (N, 3)

        ee_quat = self._quat_to_robot_root_frame(ee_quat_w)                   # (N, 4)
        cube_pos = self._world_to_robot_root_frame(cube_pos_w)                # (N, 3)
        grasp_center = self._world_to_robot_root_frame(grasp_center_w)        # (N, 3)
        rel_pos = cube_pos - grasp_center                                     # (N, 3)

        obs = torch.cat([
            joint_pos,   # 9
            ee_quat,     # 4
            cube_pos,    # 3
            rel_pos,     # 3
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Reach 阶段 reward — 单一连续函数，无分段跳变。

        使用 exp(-alpha * distance) 作为基础，distance=0 时 reward=1，远处趋近 0。
        alpha=2 保证在初始距离 ~0.5m 处仍有可感知的梯度（exp(-1)≈0.368）。
        """
        metrics  = self._get_reach_metrics()
        distance = metrics["distance"]
        success  = metrics["success"]
        # 双尺度：exp(-2d) 提供远距离梯度，exp(-10d) 在近处陡峭
        reward_reach = 0.5 * torch.exp(-2 * distance) + 0.5 * torch.exp(-10 * distance)
       
        # 线性映射：0.025（基准）→ 0，0.04（全开）→ 1，低于基准扣分，clamp 到 [0, 1]
        gripper_mean = metrics["gripper_mean"]
        reward_openning = torch.where(
            distance > self._grasp_distance_threshold,
            torch.clamp((gripper_mean - 0.025) / 0.015, -1.0, 1.0),
            torch.zeros_like(gripper_mean),
        )

        alignment_score = metrics["alignment_score"]
        reward_alignment = alignment_score

        # 线性映射：0.04(全开)→0, 0.02(挤压cube)→1, clamp [0,1]
        # cube 半宽 0.025，gripper_mean=0.025 时手指刚贴表面（零力），
        # 需要再往里挤一点到 ~0.02 才能产生 >1N 接触力触发 is_grasped
        gripper_mean = metrics["gripper_mean"]
        reward_close = torch.where(
            distance <= self._grasp_distance_threshold,
            torch.clamp((0.04 - gripper_mean) / 0.02, 0.0, 1.0),
            torch.zeros_like(gripper_mean)
        )


        is_grasped = metrics["is_grasped"]

        reward_grasp = torch.where(
            is_grasped,
            0.5,  # 固定奖励，鼓励成功抓取
            torch.zeros_like(is_grasped, dtype=torch.float),
        )

        lift_height = metrics["lift_height"]

        # lift reward：只有真正抓住 cube 才奖励抬升
        reward_lift = torch.where(
            is_grasped,
            torch.clamp(lift_height / self._lift_height_threshold, 0.0, 1.0),
            torch.zeros_like(lift_height),
        )


        phase1_reward = 0.6 * reward_reach + 0.2 * reward_openning + 0.2 * reward_alignment
        phase2_reward = 0.6 * reward_reach + 0.2 + 0.2 * reward_alignment \
                        + 0.2 * reward_close * reward_alignment + reward_grasp\
                        + 100.0 * reward_lift + 200.0 * success.float()

        reward = torch.where(
            distance > self._grasp_distance_threshold,
            phase1_reward,
            phase2_reward,
        )

        cube_pos_w = metrics["cube_pos"]                                      # (N, 3)
        cube_pos = self._world_to_robot_root_frame(cube_pos_w)                # (N, 3)
  
        # TensorBoard 日志
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update({
            "instant_success_fraction": success.float().mean(),
            
            "mean_distance": distance.mean(),
            "min_distance": distance.min(),
            "min_distance (min)": distance.min(),

            "mean_alignment_score": alignment_score.mean(),
            "mean_reward_alignment": reward_alignment.mean(),
        

            "mean_gripper_pos": gripper_mean.mean(),
            "mean_reward_openning": reward_openning.mean(),
            "mean_reward_close": reward_close.mean(),
            
            "mean_lift_height": lift_height.mean(),
            "max_lift_height (max)": lift_height.max(),
            "min_lift_height (min)": lift_height.min(),
            "mean_reward_lift": reward_lift.mean(),
            
            "mean_reward_grasp": reward_grasp.mean(),
            "grasped_fraction": is_grasped.float().mean(),

            "cube_pos_x (mean)": cube_pos[:, 0].mean(),
            "cube_pos_x (max)": cube_pos[:, 0].max(),
            "cube_pos_x (min)": cube_pos[:, 0].min(),
            "cube_pos_y (mean)": cube_pos[:, 1].mean(),
            "cube_pos_y (max)": cube_pos[:, 1].max(),
            "cube_pos_y (min)": cube_pos[:, 1].min(),
            "cube_pos_z (mean)": cube_pos[:, 2].mean(),
            "cube_pos_z (max)": cube_pos[:, 2].max(),
        })

        self.episode_success |= success 

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (terminated, truncated)。

        terminated: 进入 success_range 即结束 episode
        truncated : 超过最大步数
        """
        metrics    = self._get_reach_metrics()
        terminated = metrics["success"]
        truncated  = self.episode_length_buf >= self.max_episode_length
        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset 指定 env 的机器人和 cube 状态。"""

        # 只统计真正完成过至少一步的 episode，避免初始化阶段的 reset 污染 success rate。
        completed_env_ids = env_ids[self.episode_length_buf[env_ids] > 0] if len(env_ids) > 0 else env_ids
        if len(completed_env_ids) > 0:
            successful_episodes = int(self.episode_success[completed_env_ids].sum().item())
            self._completed_episode_count += len(completed_env_ids)
            self._successful_episode_count += successful_episodes
            self._cumulative_success_rate = self._successful_episode_count / self._completed_episode_count
    
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update({
            "episode_success_rate": torch.tensor(self._cumulative_success_rate, device=self.device),
        })


        super()._reset_idx(env_ids)

        self._reach_metrics_cache = None
        self.episode_success[env_ids] = False

        # 机器人复位到标准 home 姿态（Franka 常用初始关节角）
        default_joint_pos = torch.tensor(
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
            device=self.device,
        ).unsqueeze(0).repeat(len(env_ids), 1)

        self.robot.write_joint_state_to_sim(
            default_joint_pos,
            torch.zeros_like(default_joint_pos),
            env_ids=env_ids,
        )
        # 清零速度目标，防止旧 action 在 reset 后立即执行
        self.robot.set_joint_velocity_target(
            torch.zeros_like(default_joint_pos),
            env_ids=env_ids,
        )

        # cube 初始位置：在机器人前方桌面，加小幅随机扰动增强泛化
        cube_pos_local = torch.zeros(len(env_ids), 3, device=self.device)
        # cube_pos_local[:, 0] = 0.8 + (torch.rand(len(env_ids), device=self.device) * 0.1 - 0.05)
        # cube_pos_local[:, 1] = 0.0 + (torch.rand(len(env_ids), device=self.device) * 0.1 - 0.05)
        # cube_pos_local[:, 2] = 0.025  # 半个 cube 高度，放在桌面上
        cube_pos_local[:, 0] = 0.8
        cube_pos_local[:, 1] = 0.0
        cube_pos_local[:, 2] = 0.025  # 半个 cube 高度，放在桌面上

        cube_pos_world = cube_pos_local + self.scene.env_origins[env_ids]

        # 无旋转（w=1）
        cube_quat = torch.zeros(len(env_ids), 4, device=self.device)
        cube_quat[:, 0] = 1.0

        self.cube.write_root_pose_to_sim(
            torch.cat([cube_pos_world, cube_quat], dim=-1),
            env_ids=env_ids,
        )
        self.cube.write_root_velocity_to_sim(
            torch.zeros(len(env_ids), 6, device=self.device),
            env_ids=env_ids,
        )
