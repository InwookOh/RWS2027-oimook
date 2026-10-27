import unittest

import torch

from train import validate_training_state


class TrainingStateTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), foreach=False)
        self.model(torch.ones(2, 2)).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5, error_if_nonfinite=True, foreach=False)
        self.optimizer.step()
        self.parameter = next(self.model.parameters())

    def test_normal_state_is_accepted(self):
        validate_training_state(self.model, self.optimizer, 5)

    def test_finite_shadow_weights_without_optimizer_are_accepted(self):
        validate_training_state(self.model, None, 5)

    def test_invalid_shadow_weights_are_rejected(self):
        with torch.no_grad():
            self.parameter[0, 0] = float('inf')
        with self.assertRaisesRegex(RuntimeError, 'Non-finite model parameter'):
            validate_training_state(self.model, None, 5)

    def test_invalid_parameter_is_rejected(self):
        with torch.no_grad():
            self.parameter[0, 0] = float('nan')
        with self.assertRaisesRegex(RuntimeError, 'Non-finite model parameter'):
            validate_training_state(self.model, self.optimizer, 5)

    def test_infinite_optimizer_moment_is_rejected(self):
        self.optimizer.state[self.parameter]['exp_avg_sq'][0, 0] = float('inf')
        with self.assertRaisesRegex(RuntimeError, 'Non-finite optimizer state'):
            validate_training_state(self.model, self.optimizer, 5)

    def test_corrupted_finite_second_moment_is_rejected(self):
        self.optimizer.state[self.parameter]['exp_avg_sq'][0, 0] = 1e20
        with self.assertRaisesRegex(RuntimeError, 'outside gradient-clipping bound'):
            validate_training_state(self.model, self.optimizer, 5)


if __name__ == '__main__':
    unittest.main()
