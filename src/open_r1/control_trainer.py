"""Opt-in logging subclass for the two controls; original loss is inherited.

No additional model forward/backward passes. First two rollouts retain global
rewards/advantages and every rank's final microbatch loss inputs. Normal training
uses the unchanged GRPOTrainer. This is not a policy/KL gradient decomposition.
"""

import json
from pathlib import Path

import torch

from .advantage_controls import control_components, robust_scaled, weight_only_scaled
from .grpo_trainer import GRPOTrainer


def append_json(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")


class ControlGRPOTrainer(GRPOTrainer):
    audit_steps = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        estimator = self.advantage_estimator
        if estimator not in (robust_scaled, weight_only_scaled):
            raise ValueError("ControlGRPOTrainer requires a scale/weight control estimator")
        if self.args.token_broadcast != "uniform" or self.use_liger_loss:
            raise ValueError("Control audit expects the original uniform, non-Liger loss path")
        self.control_arm = estimator.__name__
        self.control_audit_dir = Path(self.args.output_dir) / "advantage_audit"
        self.control_audit_dir.mkdir(parents=True, exist_ok=True)
        self._control_snapshot = None

        def capture(batch, **options):
            actual = estimator(batch, **options)
            parts = control_components(batch, **options)
            mode = "train" if self.model.training else "eval"
            for name, tensor in (("final", actual), ("grpo", parts["grpo"]), ("robust", parts["robust"])):
                self._metrics[mode][f"advantage/{name}_rms"].append(tensor.double().square().mean().sqrt().item())
            self._metrics[mode]["advantage/q_mean"].append(parts["q"].mean().item())
            self._metrics[mode]["advantage/zero_group_fraction"].append(
                (actual == 0).all(1).double().mean().item())
            if mode == "train" and self.state.global_step < self.audit_steps:
                self._control_snapshot = parts[self.control_arm].flatten().detach().clone()
                if self.accelerator.is_main_process:
                    record = {"optimizer_step": self.state.global_step + 1, "arm": self.control_arm,
                              "parameters": options, "rewards": batch.rewards.detach().cpu().tolist(),
                              **{name: tensor.detach().cpu().tolist() for name, tensor in parts.items()}}
                    path = self.control_audit_dir / f"rollout_step_{self.state.global_step + 1:03d}.json"
                    path.write_text(json.dumps(record, allow_nan=False) + "\n", encoding="utf-8")
            else:
                self._control_snapshot = None
            return actual

        self.advantage_estimator = capture

    def _generate_and_score_completions(self, inputs):
        result = super()._generate_and_score_completions(inputs)
        if self._control_snapshot is not None:
            size = result["advantages"].numel()
            start = self.accelerator.process_index * size
            # These fields follow the SAME permutation/split/trim as advantages.
            result["_control_expected"] = self._control_snapshot[start:start + size]
            result["_control_row"] = torch.arange(start, start + size, device=result["advantages"].device)
        return result

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "_control_expected" in inputs:
            inputs = dict(inputs)
            expected = inputs.pop("_control_expected")
            row_ids = inputs.pop("_control_row")
            actual = inputs["advantages"]
            # Exact equality is appropriate: this checks transport, not two
            # different floating-point implementations of the same formula.
            if not torch.equal(actual, expected):
                raise ValueError("Final loss advantage changed after the control estimator")
            record = {"optimizer_step": self.state.global_step + 1,
                      "micro_step": self._step, "rank": self.accelerator.process_index,
                      "global_rollout_rows": row_ids.detach().cpu().tolist(),
                      "advantages": actual.detach().cpu().tolist(),
                      "expected_advantages": expected.detach().cpu().tolist(),
                      "valid_completion_tokens": inputs["completion_mask"].sum(1).detach().cpu().tolist(),
                      "exact_match": True}
            append_json(self.control_audit_dir / f"loss_inputs_rank_{self.accelerator.process_index}.jsonl", record)
        return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)
