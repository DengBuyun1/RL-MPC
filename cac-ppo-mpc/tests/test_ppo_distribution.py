from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from ppo_mpc.ppo import PPOAgent, PPOConfig, RolloutBuffer


class TanhNormalPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.agent = PPOAgent(
            PPOConfig(
                state_dim=7,
                hidden_dim=16,
                minibatch_size=8,
                update_epochs=1,
            ),
            device="cpu",
        )

    def test_sample_and_recomputed_log_prob_match(self):
        states = torch.randn(32, 7)
        actions, sampled_log_prob = self.agent.model.sample_action(states)
        distribution = self.agent.model.distribution(states)
        recomputed = self.agent.model.squashed_log_prob(distribution, actions=actions)
        self.assertTrue(torch.all(actions < 1.0))
        self.assertTrue(torch.all(actions > -1.0))
        self.assertTrue(torch.isfinite(sampled_log_prob).all())
        self.assertTrue(torch.allclose(sampled_log_prob, recomputed, atol=2e-5, rtol=2e-5))

    def test_saturated_tanh_uses_the_stored_bounded_action_likelihood(self):
        with torch.no_grad():
            for parameter in self.agent.model.actor.parameters():
                parameter.zero_()
            self.agent.model.actor[-1].bias.fill_(20.0)
        states = torch.zeros(4, 7)
        actions, sampled = self.agent.model.sample_action(states, deterministic=True)
        distribution = self.agent.model.distribution(states)
        recomputed = self.agent.model.squashed_log_prob(distribution, actions=actions)
        self.assertTrue(torch.all(actions < 1.0))
        self.assertTrue(torch.isfinite(sampled).all())
        self.assertTrue(torch.allclose(sampled, recomputed, atol=1e-6, rtol=1e-6))

    def test_deterministic_action_preserves_legacy_mean_mapping(self):
        states = torch.randn(8, 7)
        actions, _ = self.agent.model.sample_action(states, deterministic=True)
        expected = torch.tanh(self.agent.model.actor(states))
        self.assertTrue(torch.allclose(actions, expected, atol=1e-7, rtol=1e-7))

    def test_update_remains_finite_near_action_bounds(self):
        buffer = RolloutBuffer()
        state = np.zeros(7, dtype=np.float32)
        for index in range(16):
            action, log_prob, value = self.agent.act(state, deterministic=False)
            if index == 0:
                action = np.asarray([0.999999], dtype=np.float32)
                state_t = torch.as_tensor(state).unsqueeze(0)
                dist = self.agent.model.distribution(state_t)
                log_prob = float(
                    self.agent.model.squashed_log_prob(
                        dist, actions=torch.as_tensor(action).unsqueeze(0)
                    ).item()
                )
            buffer.add(state, action, log_prob, 0.1, False, value)
        metrics = self.agent.update(buffer, last_value=0.0)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
