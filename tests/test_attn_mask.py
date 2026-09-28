"""Attention visibility contracts for training and autoregressive generation."""

import unittest

import torch

from module import PAD_TOKEN_ID, ARPredictor


class AttentionMaskTest(unittest.TestCase):
    def _predictor(self, *, state_prediction=False, pool_grid=0):
        return ARPredictor(
            embed_dim=16,
            depth=1,
            heads=2,
            dim_head=8,
            mlp_dim=32,
            max_action_tokens=6,
            max_lang_tokens=5,
            proprio_dim=9,
            use_state_prediction=state_prediction,
            visual_pool_grid=pool_grid,
            state_pred_visual_tokens=bool(pool_grid),
        )

    def test_prefix_action_and_padding_visibility(self):
        predictor = self._predictor()
        actions = torch.tensor([[10, 20, 30, 40, PAD_TOKEN_ID, PAD_TOKEN_ID]])
        mask = predictor._build_attn_mask(
            5, torch.tensor([3]), actions, predictor.max_seq_len, torch.device("cpu")
        )
        self.assertEqual(tuple(mask.shape), (1, 1, 15, 15))
        self.assertEqual(mask.dtype, torch.bool)
        mask = mask[0, 0]
        real_prefix = [0, 1, 2, 5, 6, 7]
        for position in [3, 4, 13, 14]:
            with self.subTest(padding_position=position):
                self.assertFalse(mask[position].any())
                self.assertFalse(mask[:, position].any())
        for position in real_prefix:
            with self.subTest(prefix_position=position):
                self.assertEqual(
                    mask[position].nonzero().flatten().tolist(), real_prefix
                )
        for position in range(8, 13):
            with self.subTest(action_position=position):
                self.assertEqual(
                    mask[position].nonzero().flatten().tolist(),
                    real_prefix + list(range(8, position + 1)),
                )

    def test_state_queries_only_read_prefix_actions_and_themselves(self):
        predictor = self._predictor(state_prediction=True, pool_grid=4)
        actions = torch.tensor([[10, 20, 30, 40, PAD_TOKEN_ID, PAD_TOKEN_ID]])
        mask = predictor._build_attn_mask(
            5, torch.tensor([3]), actions, predictor.max_seq_len, torch.device("cpu")
        )[0, 0]
        # Five language slots, two 17-token views, proprio, BOS, six actions.
        action_start = 5 + 34 + 1
        query_start = action_start + 1 + 6
        real_context = [0, 1, 2] + list(range(5, action_start + 5))
        for position in range(query_start, query_start + 3):
            with self.subTest(query_position=position):
                self.assertEqual(
                    mask[position].nonzero().flatten().tolist(),
                    real_context + [position],
                )
                self.assertEqual(
                    mask[:, position].nonzero().flatten().tolist(), [position]
                )

    def test_generation_matches_training_visible_context(self):
        predictor = self._predictor(state_prediction=True, pool_grid=4)
        actions = torch.tensor([[10, 20, 30, 40, PAD_TOKEN_ID, PAD_TOKEN_ID]])
        lang_lengths = torch.tensor([3])
        train_mask = predictor._build_attn_mask(
            5, lang_lengths, actions, predictor.max_seq_len, torch.device("cpu")
        )
        visible_length = 5 + 34 + 1 + 1 + 4
        generate_mask = predictor._build_generate_mask(
            5, lang_lengths, 1, visible_length, torch.device("cpu")
        )
        torch.testing.assert_close(
            generate_mask, train_mask[:, :, :visible_length, :visible_length]
        )


if __name__ == "__main__":
    unittest.main()
