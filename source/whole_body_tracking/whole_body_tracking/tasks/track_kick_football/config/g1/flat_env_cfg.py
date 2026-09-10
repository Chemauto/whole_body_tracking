from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.track_kick_football.kick_env_cfg import KickEnvCfg


@configclass
class G1KickFlatEnvCfg(KickEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]
        # trimmed clip right_kick_trimmed.npz = right_kick.npz[170:510]:
        # reference ball-contact frame 265 -> 265 - 170 = 95
        self.commands.motion.kick_step = 95
        # default motion so the registered entry works without --motion_file
        # (train.py overrides this from CLI when provided)
        self.commands.motion.motion_file = "motions/kick_football/right_kick_trimmed.npz"
        # shorter episodes concentrate on approach + kick (motion is 6.8 s; 6.0 s
        # avoids the mid-episode motion-loop resample for late-starting envs)
        self.episode_length_s = 6.0
