"""Lightweight repository contract tests.

These checks cover configuration/script invariants that do not require LIBERO
data, GPU hardware, or external model downloads.
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractsTest(unittest.TestCase):
    def test_libero_config_declares_chunk_size(self) -> None:
        config_text = (ROOT / "config/train/data/libero.yaml").read_text()
        self.assertIn("chunk_size:", config_text)
        self.assertIn("action_dim:", config_text)
        self.assertIn("action_codec:", config_text)
        self.assertIn("type: fast", config_text)
        self.assertIn("num_bins: 256", config_text)

    def test_pretrained_vision_runner_forwards_warmup_steps_to_hydra(self) -> None:
        script_text = (ROOT / "scripts/train_eval.sh").read_text()
        self.assertIn("scheduler.warmup_steps", script_text)

    def test_runner_forwards_sp_target_space(self) -> None:
        script_text = (ROOT / "scripts/train_eval.sh").read_text()
        self.assertIn("SP_TARGET_SPACE_DEFAULT=projected", script_text)
        self.assertIn(
            'SP_TARGET_SPACE="${SP_TARGET_SPACE:-$SP_TARGET_SPACE_DEFAULT}"',
            script_text,
        )
        self.assertIn("+visual_tokens.sp_target_space", script_text)

    def test_dino768_projector_ablation_contract(self) -> None:
        config_text = (ROOT / "config/train/base.yaml").read_text()
        train_source = (ROOT / "jewam/training/train.py").read_text()
        script_text = (ROOT / "scripts/train_eval.sh").read_text()
        self.assertIn("type: mlp", config_text)
        self.assertIn("state_head_norm_type: null", config_text)
        self.assertIn(
            "projector.type='identity' requires vision hidden_dim", train_source
        )
        self.assertIn(
            "all4_sp_dinov2_frozen_visual17_patch_sp_mot_dino768_mlp", script_text
        )
        self.assertIn(
            "all4_sp_dinov2_frozen_visual17_patch_sp_mot_dino768_identity", script_text
        )
        self.assertIn('GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"', script_text)
        self.assertIn(
            '+trainer.accumulate_grad_batches="$GRAD_ACCUM_STEPS"', script_text
        )
        self.assertIn('predictor.state_head_norm_type="$STATE_HEAD_NORM"', script_text)

        mlp_start = script_text.index(
            "all4_sp_dinov2_frozen_visual17_patch_sp_mot_dino768_mlp)"
        )
        mlp_end = script_text.index(";;", mlp_start)
        self.assertIn(
            "SP_TARGET_SPACE_DEFAULT=projected",
            script_text[mlp_start:mlp_end],
        )
        identity_start = script_text.index(
            "all4_sp_dinov2_frozen_visual17_patch_sp_mot_dino768_identity)"
        )
        identity_end = script_text.index(";;", identity_start)
        self.assertIn(
            "SP_TARGET_SPACE_DEFAULT=backbone",
            script_text[identity_start:identity_end],
        )

    def test_dino192_six_layer_projector_arm_contract(self) -> None:
        config_text = (ROOT / "config/train/base.yaml").read_text()
        script_text = (ROOT / "scripts/train_eval.sh").read_text()
        self.assertIn("depth: 2", config_text)
        self.assertIn(
            "all4_sp_sigreg_dinov2_frozen_visual17_patch_sp_mot_dino192_mlp6)",
            script_text,
        )
        self.assertIn("PROJECTOR_DEPTH_DEFAULT=6", script_text)
        start = script_text.index(
            "all4_sp_sigreg_dinov2_frozen_visual17_patch_sp_mot_dino192_mlp6)"
        )
        end = script_text.index(";;", start)
        self.assertIn(
            "SP_TARGET_SPACE_DEFAULT=projected",
            script_text[start:end],
        )
        self.assertIn(
            'PROJECTOR_DEPTH="${PROJECTOR_DEPTH:-$PROJECTOR_DEPTH_DEFAULT}"',
            script_text,
        )
        self.assertIn('projector.depth="$PROJECTOR_DEPTH"', script_text)

    def test_runner_separates_checkpoint_retention_from_eval_selection(self) -> None:
        script_text = (ROOT / "scripts/train_eval.sh").read_text()
        self.assertIn('CKPT_SAVE_TOP_K="${CKPT_SAVE_TOP_K:--1}"', script_text)
        self.assertIn('+ckpt_top_k="$CKPT_SAVE_TOP_K"', script_text)
        self.assertIn('--top-k "$CKPT_SELECT_TOP_K"', script_text)

    def test_object_checkpoint_callback_can_retain_all(self) -> None:
        source = (ROOT / "jewam/training/callbacks.py").read_text()
        self.assertIn("self.top_k = int(top_k)", source)
        self.assertIn("if self.top_k == -1:", source)
        self.assertIn("return\n        while len(self._top_k_heap)", source)

    def test_pretrained_vision_runner_supports_action_only_visual17_arm(self) -> None:
        script_text = (ROOT / "scripts/train_eval.sh").read_text()
        self.assertIn("all4_dinov2_frozen_visual17)", script_text)
        self.assertIn("PRED=0", script_text)
        self.assertIn("SIGREG=0", script_text)
        self.assertIn("STATE_ARCH_DEFAULT=shared", script_text)
        self.assertIn("POOL_GRID_DEFAULT=4", script_text)
        self.assertIn("PATCH_SP_DEFAULT=false", script_text)

    def test_pretrained_vision_runner_supports_action_only_visual17_mot_arm(
        self,
    ) -> None:
        script_text = (ROOT / "scripts/train_eval.sh").read_text()
        self.assertIn("all4_dinov2_frozen_visual17_mot)", script_text)
        self.assertIn("STATE_ARCH_DEFAULT=mot", script_text)

    def test_saved_fast_tokenizer_patch_copies_processor_module_fallback(self) -> None:
        source = (ROOT / "jewam/data/preprocessing.py").read_text()
        self.assertIn("tokenizer: Any | None = None", source)
        self.assertIn("inspect.getfile", source)
        self.assertIn("tokenizer.__class__.__name__", source)
        self.assertIn("processing_action_tokenizer.py", source)

    def test_weights_loader_rejects_visual_prefix_checkpoint(self) -> None:
        source = (ROOT / "jewam/evaluation/libero.py").read_text()
        self.assertIn("view_embedding", source)
        self.assertIn("agent_patch_2d_pos", source)
        self.assertIn("visual-prefix", source)

    def test_predict_actions_defaults_to_predictor_token_budget(self) -> None:
        source = (ROOT / "jewam/models/policy.py").read_text()
        self.assertIn("max_len=None", source)
        self.assertIn("self.predictor.max_action_tokens", source)

    def test_language_instruction_is_not_silently_empty(self) -> None:
        preprocess_source = (ROOT / "jewam/data/preprocessing.py").read_text()
        dataset_source = (ROOT / "jewam/data/libero.py").read_text()
        self.assertIn("_derive_instruction_from_filename", preprocess_source)
        self.assertIn("language_source", preprocess_source)
        self.assertIn("Empty language_instruction", dataset_source)
        self.assertIn("use_language=False", dataset_source)

    def test_gripper_aux_is_denormalized_before_eval_execution(self) -> None:
        source = (ROOT / "jewam/evaluation/libero.py").read_text()
        self.assertIn("_denormalize_gripper_aux", source)
        self.assertIn("action_low[6]", source)
        self.assertIn("action_high[6]", source)

    def test_denormalize_actions_keeps_zero_range_dims_constant(self) -> None:
        source = (ROOT / "jewam/actions/fast.py").read_text()
        self.assertIn("zero_range = half_range < 1e-8", source)
        self.assertIn("np.where(zero_range, mid, restored)", source)

    def test_preprocess_writes_generic_action_token_contract(self) -> None:
        source = (ROOT / "jewam/data/preprocessing.py").read_text()
        self.assertIn('"action_tokens"', source)
        self.assertIn('"action_length"', source)
        self.assertIn('"action_codec_type"', source)
        self.assertIn('"action_vocab_size"', source)
        self.assertIn('"action_token_min"', source)
        self.assertIn('"action_token_max"', source)

    def test_dataset_falls_back_to_legacy_fast_token_fields(self) -> None:
        source = (ROOT / "jewam/data/libero.py").read_text()
        self.assertIn('"action_tokens"', source)
        self.assertIn('"action_length"', source)
        self.assertIn('"fast_tokens"', source)
        self.assertIn('"fast_length"', source)

    def test_preprocess_runner_validates_codec_before_skipping_existing_outputs(
        self,
    ) -> None:
        source = (ROOT / "scripts/preprocess_all4.sh").read_text()
        self.assertIn("check_existing_codec", source)
        self.assertIn("action_codec_type", source)
        self.assertIn("ACTION_CODEC", source)

    def test_training_seed_is_used_globally(self) -> None:
        source = (ROOT / "jewam/training/train.py").read_text()
        self.assertIn("seed = int(cfg.seed)", source)
        self.assertIn("pl.seed_everything(seed, workers=True)", source)
        self.assertIn("seed=seed", source)

    def test_eval_scripts_use_run_seed(self) -> None:
        scripts = ["scripts/train_eval.sh"]
        for script in scripts:
            with self.subTest(script=script):
                source = (ROOT / script).read_text()
                self.assertNotIn("--seed 42", source)
                self.assertIn('--seed "$SEED"', source)

    def test_libero_eval_defaults_match_openvla_contract(self) -> None:
        eval_source = (ROOT / "jewam/evaluation/libero.py").read_text()
        runner_source = (ROOT / "scripts/train_eval.sh").read_text()
        self.assertIn("OPENVLA_NUM_TRIALS_PER_TASK = 50", eval_source)
        self.assertIn("OPENVLA_DUMMY_WAIT_STEPS = 10", eval_source)
        for suite, max_steps in {
            "libero_spatial": 220,
            "libero_object": 280,
            "libero_goal": 300,
            "libero_10": 520,
            "libero_90": 400,
        }.items():
            self.assertIn(f'"{suite}": {max_steps}', eval_source)
            self.assertIn(f"{suite}) echo {max_steps}", runner_source)
        self.assertNotIn("LIGHT_EVAL_EPISODES", runner_source)
        self.assertNotIn("LIGHT_EVAL_MAX_STEPS", runner_source)
        self.assertNotIn("select_light_eval_ckpt.py", runner_source)
        self.assertNotIn("light eval", runner_source)
        self.assertIn(
            'EVAL_LOG="${CKPT_DIR}/eval_${candidate_stem}_${suite}.log"', runner_source
        )
        self.assertIn("[JEWAM] eval candidate:", runner_source)
        self.assertIn('FINAL_EVAL_EPISODES="${FINAL_EVAL_EPISODES:-50}"', runner_source)
        self.assertIn(
            'FINAL_EVAL_MAX_STEPS="${FINAL_EVAL_MAX_STEPS:-openvla}"', runner_source
        )
        self.assertIn('EVAL_NUM_STEPS_WAIT="${EVAL_NUM_STEPS_WAIT:-10}"', runner_source)
        self.assertIn('--num-steps-wait "$EVAL_NUM_STEPS_WAIT"', runner_source)

    def test_libero_eval_success_contract_uses_done_and_failures(self) -> None:
        eval_source = (ROOT / "jewam/evaluation/libero.py").read_text()
        self.assertIn("def _dummy_wait(", eval_source)
        self.assertIn("success = bool(done)", eval_source)
        self.assertIn("except Exception as exc", eval_source)
        self.assertIn('failure_reason = f"exception: {exc}"', eval_source)
        self.assertIn('failure_reason = "max_steps_exceeded"', eval_source)

    def test_removed_legacy_entrypoints_stay_removed(self) -> None:
        removed = [
            "eval.py",
            "smoke_test.py",
            "run_all4.sh",
            "run_ablation.sh",
            "run_visual17.sh",
            "slides",
            "config/eval",
            "CLAUDE.md",
        ]
        for relpath in removed:
            with self.subTest(relpath=relpath):
                self.assertFalse((ROOT / relpath).exists())


if __name__ == "__main__":
    unittest.main()
