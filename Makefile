.DEFAULT_GOAL := help
.PHONY: help doctor doctor-cuda env-create download-models cache smoke train dry-run logs plot-training validate-run install style quality test slow_test evaluate

PYTHON ?= python
GRPO := $(PYTHON) scripts/grpo.py

help:
	@$(GRPO) --help
	@echo 'Make shortcuts: env-create install doctor doctor-cuda download-models cache smoke train dry-run logs validate-run plot-training test'
	@echo 'Default GPUs: VLLM_GPUS=4, QRM_GPU=5, TRAIN_GPUS=6,7; set DATASET_NAME before launch.'

doctor:
	$(GRPO) doctor

doctor-cuda:
	$(GRPO) doctor --cuda

env-create:
	conda env create -f environment.yml

download-models:
	$(GRPO) download

cache:
	$(GRPO) cache

smoke:
	$(GRPO) smoke

train:
	$(GRPO) train

dry-run:
	$(GRPO) smoke --dry-run

logs:
	$(GRPO) logs --follow

validate-run:
	$(GRPO) validate $(if $(RUN_DIR),--run-dir $(RUN_DIR),) $(if $(REQUIRE_MERGED),--require-merged,)

plot-training:
	$(PYTHON) scripts/plot_training_metrics.py $(if $(RUN_DIR),$(RUN_DIR),)

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := src tests


# dev dependencies
install:
	$(PYTHON) -m pip install -e '.[judge]'
	$(PYTHON) -m pip install flash-attn==2.7.4.post1 --no-build-isolation
	$(PYTHON) -m pip check

style:
	ruff format --line-length 119 --target-version py310 $(check_dirs) setup.py
	isort $(check_dirs) setup.py

quality:
	ruff check --line-length 119 --target-version py310 $(check_dirs) setup.py
	isort --check-only $(check_dirs) setup.py
	flake8 --max-line-length 119 $(check_dirs) setup.py

test:
	$(PYTHON) -m pytest -q --ignore=tests/slow/ tests/

slow_test:
	python -m pytest -sv -vv tests/slow/

# Evaluation

evaluate:
	$(eval PARALLEL_ARGS := $(if $(PARALLEL),$(shell \
		if [ "$(PARALLEL)" = "data" ]; then \
			echo "data_parallel_size=$(NUM_GPUS)"; \
		elif [ "$(PARALLEL)" = "tensor" ]; then \
			echo "tensor_parallel_size=$(NUM_GPUS)"; \
		fi \
	),))
	$(if $(filter tensor,$(PARALLEL)),export VLLM_WORKER_MULTIPROC_METHOD=spawn &&,) \
	MODEL_ARGS="pretrained=$(MODEL),dtype=bfloat16,$(PARALLEL_ARGS),max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" && \
	if [ "$(TASK)" = "lcb" ]; then \
		lighteval vllm $$MODEL_ARGS "extended|lcb:codegeneration|0|0" \
			--use-chat-template \
			--output-dir data/evals/$(MODEL); \
	else \
		lighteval vllm $$MODEL_ARGS "lighteval|$(TASK)|0|0" \
			--use-chat-template \
			--output-dir data/evals/$(MODEL); \
	fi
