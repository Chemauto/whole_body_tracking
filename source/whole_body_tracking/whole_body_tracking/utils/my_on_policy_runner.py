import os

from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MotionOnPolicyRunner(OnPolicyRunner):
    """On-policy runner that also exports the policy to ONNX on every checkpoint save.

    The ONNX policy (with deployment metadata attached) is written next to the model
    checkpoint so it can be used directly for sim-to-real deployment without any cloud
    logger dependency.
    """

    def save(self, path: str, infos=None):
        """Save the model, training information, and an ONNX copy of the policy."""
        super().save(path, infos)
        policy_path = path.split("model")[0]
        filename = policy_path.split("/")[-2] + ".onnx"
        export_motion_policy_as_onnx(
            self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
        )
        run_label = os.path.basename(os.path.normpath(self.log_dir)) if self.log_dir else "local"
        attach_onnx_metadata(self.env.unwrapped, run_label, path=policy_path, filename=filename)
        print(f"[INFO]: Exported ONNX policy to: {policy_path}{filename}")
