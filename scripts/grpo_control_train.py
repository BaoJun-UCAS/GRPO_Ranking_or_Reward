#!/usr/bin/env python3
"""Use the normal training main with the opt-in control audit subclass."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_r1 import grpo
from open_r1.control_trainer import ControlGRPOTrainer


if __name__ == "__main__":
    # Only this dedicated process uses the subclass; the original module/file
    # and all other launchers retain their usual trainer and objective.
    grpo.GRPOTrainer = ControlGRPOTrainer
    parser = grpo.TrlParser((grpo.GRPOScriptArguments, grpo.GRPOConfig, grpo.ModelConfig))
    grpo.main(*parser.parse_args_and_config())
