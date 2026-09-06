"""Temporal encoder-decoder policy for the kick task (arXiv:2511.03996 style).

The plain kick policy is a single-frame MLP: when the virtual perception
system loses the ball, the actor is blind. This module adds short-term memory:

  actor obs = [current obs (incl. virtual ball obs) | flattened 1 s ball history]
                                        |
              hist_encoder MLP --> 64-d latent --\
              decoder MLP --> privileged ball state (train-time only)
                                        |
  actor input = [current obs | latent]  (decoder is dropped at deployment)

The decoder is supervised from privileged critic-observation channels (true
ball state + velocity) stored in the rollout buffer, so the latent is shaped
to be an implicit state estimate rather than a free embedding.

Kept separate from the stock MotionOnPolicyRunner path: training uses this
module only when --temporal_actor is passed to train.py.
"""

from __future__ import annotations

import torch

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.networks import MLP
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner


def make_temporal_actor_critic(base_class):
    """Compose a temporal variant from a stock ActorCritic subclass.

    Keeps rsl_rl's ActorCritic stock behaviour (normalizers, distribution,
    evaluate, ...) and only changes how the actor observation is assembled:
    the raw history block is replaced by its latent code.
    """

    class TemporalActorCritic(base_class):
        def __init__(
            self,
            obs,
            obs_groups,
            num_actions,
            hist_start: int,
            hist_len: int,
            ball_dim: int = 5,
            latent_dim: int = 64,
            encoder_hidden_dims: tuple[int, ...] = (256,),
            decoder_out_dim: int = 6,
            decoder_hidden_dims: tuple[int, ...] = (128,),
            decoder_target_slice: tuple[int, int] = (0, 0),
            **kwargs,
        ):
            self.hist_start = hist_start
            self.hist_len = hist_len
            self.ball_dim = ball_dim
            self.latent_dim = latent_dim
            # let the parent build actor/critic on the effective input
            # [current(...) | latent] by shrinking the history block
            shrunk_obs = obs.clone()
            for group in obs_groups["policy"]:
                full = obs[group]
                shrunk = torch.cat(
                    [
                        full[:, :hist_start],
                        torch.zeros(full.shape[0], latent_dim, device=full.device),
                        full[:, hist_start + hist_len :],
                    ],
                    dim=-1,
                )
                shrunk_obs[group] = shrunk
            super().__init__(shrunk_obs, obs_groups, num_actions, **kwargs)

            self.hist_encoder = MLP(hist_len, latent_dim, list(encoder_hidden_dims), "elu")
            # train-time only: reconstructs privileged ball state from the latent
            self.decoder = MLP(latent_dim, decoder_out_dim, list(decoder_hidden_dims), "elu")
            self.decoder_target_slice = slice(*decoder_target_slice)
            self.decoder_critic_group = obs_groups["critic"][0]

        def _embed(self, actor_obs: torch.Tensor) -> torch.Tensor:
            hist = actor_obs[:, self.hist_start : self.hist_start + self.hist_len]
            latent = self.hist_encoder(hist)
            return torch.cat([actor_obs[:, : self.hist_start], latent, actor_obs[:, self.hist_start + self.hist_len :]], dim=-1)

        def get_actor_obs(self, obs) -> torch.Tensor:
            return self._embed(super().get_actor_obs(obs))

    return TemporalActorCritic


class PPOWithDecoder(PPO):
    """PPO plus an auxiliary decoder pass that shapes the history latent.

    The policy loss already backpropagates through the encoder (it is part of
    the actor's forward path); the decoder pass adds a supervised gradient that
    forces the latent to reconstruct the true ball state from noisy history --
    the implicit state estimation of arXiv:2511.03996.
    """

    def __init__(self, *args, aux_learning_rate: float = 1e-4, aux_loss_coef: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        params = list(self.policy.hist_encoder.parameters()) + list(self.policy.decoder.parameters())
        self.aux_optimizer = torch.optim.Adam(params, lr=aux_learning_rate)
        self.aux_loss_coef = aux_loss_coef

    def update(self) -> dict[str, float]:
        stats = super().update()
        total, count = 0.0, 0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, num_epochs=1):
            obs_batch = batch[0]
            policy_group = self.policy.obs_groups["policy"][0]
            actor_obs = obs_batch[policy_group]
            critic_obs = obs_batch[self.policy.decoder_critic_group]
            hist = actor_obs[:, self.policy.hist_start : self.policy.hist_start + self.policy.hist_len]
            pred = self.policy.decoder(self.policy.hist_encoder(hist))
            target = critic_obs[:, self.policy.decoder_target_slice]
            loss = self.aux_loss_coef * torch.nn.functional.mse_loss(pred, target)
            self.aux_optimizer.zero_grad()
            loss.backward()
            self.aux_optimizer.step()
            total += loss.item()
            count += 1
        stats["loss/decoder"] = total / max(count, 1)
        return stats


class KickTemporalOnPolicyRunner(MotionOnPolicyRunner):
    """Runner wiring TemporalActorCritic + PPOWithDecoder, dimensions derived from the env.

    Actor observation layout (fixed by kick_env_cfg):
      [command(2*num_joints) | ball_virtual(5) | history(N*5) | rest]
    The encoder consumes the history block; everything else flows straight
    through, so stage-1 warm start keeps channel semantics.
    """

    BALL_DIM = 5  # [est_pos(2), dir(2), visible(1)]

    def _construct_algorithm(self, obs):
        from rsl_rl.modules.actor_critic import ActorCritic

        env = self.env.unwrapped
        command = env.command_manager.get_term("motion")
        prefix = 2 * env.scene["robot"].num_joints
        hist_len = command.cfg.ball_history_length * self.BALL_DIM
        hist_start = prefix + self.BALL_DIM

        self.policy_cfg.pop("class_name", None)
        policy_class = make_temporal_actor_critic(ActorCritic)
        policy = policy_class(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            hist_start=hist_start,
            hist_len=hist_len,
            ball_dim=self.BALL_DIM,
            decoder_target_slice=(prefix, prefix + 6),  # true ball state(4) + velocity(2)
            **self.policy_cfg,
        ).to(self.device)

        self.alg_cfg.pop("class_name", None)
        alg = PPOWithDecoder(policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)
        alg.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return alg

    def save(self, path: str, infos=None) -> None:
        # plain checkpoint save: the ONNX exporter assumes a flat actor MLP and
        # is not yet adapted to the encoder-decoder policy (deployment TODO)
        OnPolicyRunner.save(self, path, infos)
