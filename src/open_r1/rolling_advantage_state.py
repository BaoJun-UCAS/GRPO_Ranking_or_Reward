"""Persist adaptive history alongside the exact checkpoint that consumes it."""

from pathlib import Path

from transformers import TrainerCallback


class RollingAdvantageStateCallback(TrainerCallback):
    def __init__(self, estimator):
        self.estimator = estimator

    def on_train_begin(self, args, state, control, **kwargs):
        saved_step = self.estimator.state_dict()["step"]
        if state.global_step > 0 and (not self.estimator.resume_loaded or saved_step is None
                                      or self.estimator.checkpoint_global_step != state.global_step
                                      or saved_step >= state.global_step):
            raise ValueError("Resumed optimizer step does not match the restored rolling advantage history")
        if state.global_step == 0 and saved_step is not None:
            raise ValueError("A fresh training run requires a fresh rolling advantage estimator")

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero and args.should_save:
            self.estimator.save(Path(args.output_dir) / f"checkpoint-{state.global_step}",
                                global_step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and args.should_save:
            self.estimator.save(args.output_dir, global_step=state.global_step)
