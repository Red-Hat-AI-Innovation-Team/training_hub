"""Speculative decoding draft model training via the speculators library.

Supports Eagle3, DFlash, MTP, and PEagle speculator architectures for
accelerating LLM inference through speculative decoding.

Pipeline stages (offline mode):
    1. prepare_data  -- tokenize dataset, build loss masks, compute token freqs
    2. launch_vllm   -- start vLLM with extract_hidden_states config
    3. generate_hs   -- extract hidden states from verifier model via vLLM
    4. train         -- train draft model on hidden states

Usage::

    from training_hub import train_speculator, eagle3

    # Full offline pipeline
    result = train_speculator(
        verifier_name_or_path="Qwen/Qwen3-8B",
        data_path="sharegpt",
        ckpt_output_dir="./eagle3_output",
        epochs=3,
    )

    # Convenience alias for Eagle3
    result = eagle3("Qwen/Qwen3-8B", "./output", data_path="sharegpt")
"""
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from . import Algorithm, AlgorithmRegistry, Backend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class SpeculatorsBackend(Backend):
    """Speculators library backend for speculative decoding training.

    Wraps the speculators offline/online training pipeline and manages
    the vLLM server lifecycle for hidden state extraction.
    """

    def execute_training(self, algorithm_params: Dict[str, Any]) -> Any:
        try:
            import speculators  # noqa: F401
        except ImportError:
            raise ImportError(
                "Speculative decoding requires the 'speculators' package. "
                "Install with: pip install speculators"
            ) from None

        mode = algorithm_params.get("mode", "offline")

        if mode == "offline":
            return self._run_offline_pipeline(algorithm_params)
        elif mode == "online":
            return self._run_online_pipeline(algorithm_params)
        elif mode == "train_only":
            return self._run_train_only(algorithm_params)
        elif mode == "data_only":
            return self._run_data_only(algorithm_params)
        else:
            raise ValueError(
                f"Unknown mode: '{mode}'. "
                "Must be one of: offline, online, train_only, data_only"
            )

    # -- Pipeline modes -------------------------------------------------------

    def _run_offline_pipeline(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Full offline pipeline: prepare data -> gen hidden states -> train."""
        t_start = time.time()
        stage_times: Dict[str, float] = {}

        verifier = params["verifier_name_or_path"]
        ckpt_dir = params["ckpt_output_dir"]
        data_output_dir = params.get("data_output_dir") or os.path.join(ckpt_dir, "data")
        hidden_states_path = params.get("hidden_states_path") or os.path.join(data_output_dir, "hidden_states")

        os.makedirs(data_output_dir, exist_ok=True)
        os.makedirs(hidden_states_path, exist_ok=True)

        # Stage 1: Prepare data (skip if already done)
        t0 = time.time()
        arrow_marker = os.path.join(data_output_dir, "dataset_info.json")
        if not os.path.exists(arrow_marker):
            self._prepare_data(params, verifier, data_output_dir)
        else:
            logger.info("Skipping data prep — Arrow dataset already exists at %s", data_output_dir)
        stage_times["prepare_data"] = time.time() - t0
        logger.info("Data preparation complete (%.1fs)", stage_times["prepare_data"])

        # Stage 2+3: Generate hidden states (launch vLLM, run extraction, stop vLLM)
        t0 = time.time()
        vllm_endpoint = params.get("vllm_endpoint")
        managed_proc = None
        try:
            if not vllm_endpoint:
                managed_proc, vllm_endpoint = self._launch_vllm(params, verifier, hidden_states_path)

            self._generate_hidden_states(
                params, vllm_endpoint, data_output_dir, hidden_states_path
            )
        finally:
            if managed_proc is not None:
                self._stop_vllm(managed_proc)
        stage_times["hidden_state_gen"] = time.time() - t0
        logger.info("Hidden state generation complete (%.1fs)", stage_times["hidden_state_gen"])

        # Stage 4: Train
        t0 = time.time()
        train_result = self._run_training(
            params, verifier, data_output_dir, hidden_states_path, ckpt_dir
        )
        stage_times["training"] = time.time() - t0
        logger.info("Training complete (%.1fs)", stage_times["training"])

        return {
            "status": "success",
            "mode": "offline",
            "checkpoint_path": ckpt_dir,
            "total_time_seconds": time.time() - t_start,
            "stage_times": stage_times,
            "data_output_dir": data_output_dir,
            "hidden_states_path": hidden_states_path,
            **train_result,
        }

    def _run_online_pipeline(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Online mode: prepare data, then train with on-demand hidden state gen."""
        t_start = time.time()

        verifier = params["verifier_name_or_path"]
        ckpt_dir = params["ckpt_output_dir"]
        data_output_dir = params.get("data_output_dir") or os.path.join(ckpt_dir, "data")
        hidden_states_path = params.get("hidden_states_path") or os.path.join(data_output_dir, "hidden_states")

        os.makedirs(data_output_dir, exist_ok=True)
        os.makedirs(hidden_states_path, exist_ok=True)

        # Prepare data (skip if Arrow dataset already exists)
        arrow_marker = os.path.join(data_output_dir, "dataset_info.json")
        if not os.path.exists(arrow_marker):
            self._prepare_data(params, verifier, data_output_dir)
        else:
            logger.info("Skipping data prep — Arrow dataset already exists at %s", data_output_dir)

        # Launch managed vLLM if no endpoint provided
        vllm_endpoint = params.get("vllm_endpoint")
        managed_proc = None
        try:
            if not vllm_endpoint:
                managed_proc, vllm_endpoint = self._launch_vllm(params, verifier, hidden_states_path)

            # Train with on_missing="generate"
            train_result = self._run_training(
                params, verifier, data_output_dir, hidden_states_path, ckpt_dir,
                on_missing="generate",
                vllm_endpoint=vllm_endpoint,
            )
        finally:
            if managed_proc is not None:
                self._stop_vllm(managed_proc)

        return {
            "status": "success",
            "mode": "online",
            "checkpoint_path": ckpt_dir,
            "total_time_seconds": time.time() - t_start,
            **train_result,
        }

    def _run_train_only(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Train from pre-existing hidden states on disk."""
        t_start = time.time()

        verifier = params["verifier_name_or_path"]
        ckpt_dir = params["ckpt_output_dir"]
        data_output_dir = params.get("data_output_dir")
        hidden_states_path = params.get("hidden_states_path")

        if not data_output_dir:
            raise ValueError("train_only mode requires 'data_output_dir' (path to Arrow dataset)")
        if not hidden_states_path:
            raise ValueError("train_only mode requires 'hidden_states_path' (path to hs_*.safetensors)")

        train_result = self._run_training(
            params, verifier, data_output_dir, hidden_states_path, ckpt_dir,
            on_missing="raise",
        )

        return {
            "status": "success",
            "mode": "train_only",
            "checkpoint_path": ckpt_dir,
            "total_time_seconds": time.time() - t_start,
            **train_result,
        }

    def _run_data_only(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Data prep + hidden state generation only, no training."""
        t_start = time.time()
        stage_times: Dict[str, float] = {}

        verifier = params["verifier_name_or_path"]
        ckpt_dir = params["ckpt_output_dir"]
        data_output_dir = params.get("data_output_dir") or os.path.join(ckpt_dir, "data")
        hidden_states_path = params.get("hidden_states_path") or os.path.join(data_output_dir, "hidden_states")

        os.makedirs(data_output_dir, exist_ok=True)
        os.makedirs(hidden_states_path, exist_ok=True)

        # Prepare data
        t0 = time.time()
        self._prepare_data(params, verifier, data_output_dir)
        stage_times["prepare_data"] = time.time() - t0

        # Generate hidden states
        t0 = time.time()
        vllm_endpoint = params.get("vllm_endpoint")
        managed_proc = None
        try:
            if not vllm_endpoint:
                managed_proc, vllm_endpoint = self._launch_vllm(params, verifier, hidden_states_path)
            self._generate_hidden_states(
                params, vllm_endpoint, data_output_dir, hidden_states_path
            )
        finally:
            if managed_proc is not None:
                self._stop_vllm(managed_proc)
        stage_times["hidden_state_gen"] = time.time() - t0

        return {
            "status": "data_ready",
            "mode": "data_only",
            "data_output_dir": data_output_dir,
            "hidden_states_path": hidden_states_path,
            "total_time_seconds": time.time() - t_start,
            "stage_times": stage_times,
        }

    # -- Stage implementations ------------------------------------------------

    def _prepare_data(
        self,
        params: Dict[str, Any],
        verifier: str,
        data_output_dir: str,
    ) -> None:
        """Stage 1: Tokenize dataset, build loss masks, compute token freqs."""
        from speculators.data_generation.preprocessing import (
            load_and_preprocess_dataset,
        )

        data_path = params.get("data_path")
        if not data_path:
            raise ValueError("'data_path' is required for data preparation.")

        # Resolve data paths -- speculators expects a list
        if data_path in ("sharegpt",):
            # Built-in dataset name handled by speculators
            train_data_paths = [data_path]
        elif os.path.isfile(data_path):
            train_data_paths = [data_path]
        else:
            # Assume HF dataset ID
            train_data_paths = [data_path]

        seq_length = params.get("total_seq_len", 2048)
        max_samples = params.get("max_samples")
        trust_remote_code = params.get("trust_remote_code", False)
        token_freq_path = os.path.join(data_output_dir, "token_freq.pt")

        dataset, processor = load_and_preprocess_dataset(
            target_model_path=verifier,
            train_data_paths=train_data_paths,
            seq_length=seq_length,
            max_samples=max_samples,
            token_freq_path=token_freq_path,
            trust_remote_code=trust_remote_code,
        )

        # Save Arrow dataset to disk
        dataset.save_to_disk(data_output_dir)
        logger.info("Saved preprocessed dataset to %s (%d samples)", data_output_dir, len(dataset))

    def _launch_vllm(
        self,
        params: Dict[str, Any],
        verifier: str,
        hidden_states_path: str,
    ) -> tuple:
        """Launch a managed vLLM server with extract_hidden_states config.

        Returns (process, endpoint_url).
        """
        port = self._find_free_port()
        vllm_gpu_ids = params.get("vllm_gpu_ids")
        num_gpus = params.get("num_gpus", 1)
        if vllm_gpu_ids:
            num_gpus = len(vllm_gpu_ids)
        gpu_mem_util = params.get("vllm_gpu_memory_utilization", 0.9)
        trust_remote_code = params.get("trust_remote_code", False)

        # Resolve target layer IDs
        target_layer_ids = params.get("target_layer_ids")
        if not target_layer_ids:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(verifier, trust_remote_code=trust_remote_code)
            if hasattr(config, "text_config"):
                config = config.text_config
            n_layers = config.num_hidden_layers
            target_layer_ids = [2, n_layers // 2, n_layers - 3, n_layers]

        speculative_config = {
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": target_layer_ids}
            },
        }
        kv_transfer_config = {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": hidden_states_path},
        }

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", verifier,
            "--speculative_config", json.dumps(speculative_config),
            "--kv_transfer_config", json.dumps(kv_transfer_config),
            "--port", str(port),
            "--gpu-memory-utilization", str(gpu_mem_util),
            "--no-enable-chunked-prefill",
        ]

        if num_gpus > 1:
            cmd.extend(["--data-parallel-size", str(num_gpus)])

        if trust_remote_code:
            cmd.append("--trust-remote-code")

        # Set GPU visibility for managed vLLM subprocess
        env = os.environ.copy()
        if vllm_gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in vllm_gpu_ids)
            logger.info("vLLM GPUs: %s", env["CUDA_VISIBLE_DEVICES"])

        logger.info("Launching vLLM server: %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )

        endpoint = f"http://localhost:{port}/v1"
        health_url = f"http://localhost:{port}/health"
        self._wait_for_server(health_url, timeout=params.get("vllm_startup_timeout", 600))
        logger.info("vLLM server ready at %s", endpoint)

        # Warmup request to trigger model compilation before training starts
        self._warmup_vllm(endpoint, verifier)

        return proc, endpoint

    @staticmethod
    def _warmup_vllm(endpoint: str, model: str) -> None:
        """Send a small warmup request to trigger vLLM model compilation."""
        import urllib.request

        logger.info("Sending warmup request to vLLM...")
        payload = json.dumps({
            "model": model,
            "prompt": "Hello",
            "max_tokens": 1,
        }).encode()
        req = urllib.request.Request(
            f"{endpoint}/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=300)
            logger.info("vLLM warmup complete")
        except Exception as e:
            logger.warning("vLLM warmup request failed (may still work): %s", e)

    @staticmethod
    def _find_free_port() -> int:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    @staticmethod
    def _wait_for_server(health_url: str, timeout: int = 600) -> None:
        """Poll health endpoint until server is ready."""
        import urllib.request
        import urllib.error

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                req = urllib.request.urlopen(health_url, timeout=5)
                if req.status == 200:
                    return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)
        raise TimeoutError(
            f"vLLM server did not become ready within {timeout}s. "
            "Check GPU availability and model loading."
        )

    @staticmethod
    def _stop_vllm(proc: subprocess.Popen) -> None:
        """Terminate a managed vLLM server process."""
        if proc.poll() is not None:
            return
        logger.info("Stopping managed vLLM server (pid=%d)", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    def _generate_hidden_states(
        self,
        params: Dict[str, Any],
        vllm_endpoint: str,
        data_output_dir: str,
        hidden_states_path: str,
    ) -> None:
        """Stage 3: Extract hidden states from verifier via vLLM."""
        # Run the data_generation_offline script as a subprocess since it uses
        # asyncio internally and is designed as a standalone entry point.
        max_samples = params.get("max_samples")
        concurrency = params.get("datagen_concurrency", 32)

        cmd = [
            sys.executable, "-m", "speculators",
        ]

        # Fall back to running the script directly since speculators may not
        # expose data_generation_offline as a CLI subcommand.
        # Use the scripts path from the speculators package.
        import speculators as _spec
        spec_root = Path(_spec.__file__).parent.parent.parent  # src/speculators -> src -> package root
        script_path = spec_root / "scripts" / "data_generation_offline.py"

        if not script_path.exists():
            # Try installed package location
            import importlib.resources
            # Fall back to a direct import approach
            script_path = None

        cmd = [sys.executable]
        if script_path and script_path.exists():
            cmd.append(str(script_path))
        else:
            # Use the module path relative to speculators install
            cmd.extend(["-c", (
                "from speculators.data_generation.offline import *; "
                "raise NotImplementedError('Direct import not available, use scripts/')"
            )])
            raise RuntimeError(
                "Could not locate speculators data_generation_offline.py script. "
                "Ensure speculators is installed from source or the scripts/ directory "
                "is accessible."
            )

        cmd.extend([
            "--preprocessed-data", data_output_dir,
            "--endpoint", vllm_endpoint,
            "--output", hidden_states_path,
            "--concurrency", str(concurrency),
            "--validate-outputs",
        ])
        if max_samples:
            cmd.extend(["--max-samples", str(max_samples)])

        logger.info("Generating hidden states: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Hidden state generation failed (exit code {result.returncode}):\n"
                f"{result.stderr[-2000:] if result.stderr else result.stdout[-2000:]}"
            )

        # Count generated files
        hs_count = len(list(Path(hidden_states_path).glob("hs_*.safetensors")))
        logger.info("Generated %d hidden state files in %s", hs_count, hidden_states_path)

    def _run_training(
        self,
        params: Dict[str, Any],
        verifier: str,
        data_output_dir: str,
        hidden_states_path: str,
        ckpt_dir: str,
        on_missing: str = "raise",
        vllm_endpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stage 4: Train the draft model using speculators' Trainer."""
        import argparse

        import torch
        from transformers import AutoConfig

        from speculators.model import SpeculatorModel
        from speculators.models.eagle3.data import shift_batch
        from speculators.models.metrics import resolve_loss_fn
        from speculators.train.data import ArrowDataset, create_collate_fn
        from speculators.train.distributed_batch_sampler import (
            MultipackDistributedBatchSamplerV2,
        )
        from speculators.train.noise_transforms import AddUniformNoise
        from speculators.train.trainer import Trainer, TrainerConfig
        from speculators.train.utils import resolve_mask_token_id
        from speculators.train.vocab_mapping import (
            build_vocab_mappings_from_distribution,
            get_target_vocab_size,
        )

        # -- Resolve parameters with defaults --
        speculator_type = params.get("speculator_type", "eagle3")
        epochs = params.get("epochs", 3)
        lr = params.get("lr", 1e-4)
        total_seq_len = params.get("total_seq_len", 2048)
        draft_vocab_size = params.get("draft_vocab_size")
        num_layers = params.get("num_layers", 1)
        ttt_steps = params.get("ttt_steps", 3)
        target_layer_ids = params.get("target_layer_ids")
        noise_std = params.get("noise_std", 0.05)
        loss_fn_name = params.get("loss_fn", "kl_div")
        scheduler_type = params.get("scheduler_type", "linear")
        checkpoint_freq = params.get("checkpoint_freq", 1.0)
        log_freq = params.get("log_freq", 1)
        num_workers = params.get("num_workers", 12)
        trust_remote_code = params.get("trust_remote_code", False)
        hidden_states_dtype = getattr(torch, params.get("hidden_states_dtype", "bfloat16"))
        from_pretrained = params.get("from_pretrained", "")

        # -- Vocab mappings --
        d2t, t2d = None, None
        token_freq_path = Path(data_output_dir) / "token_freq.pt"
        d2t_path = Path(data_output_dir) / "d2t.npy"
        t2d_path = Path(data_output_dir) / "t2d.npy"

        if d2t_path.exists() and t2d_path.exists():
            import numpy as np
            d2t = torch.from_numpy(np.load(d2t_path))
            t2d = torch.from_numpy(np.load(t2d_path))
            draft_vocab_size = d2t.shape[0]
        elif token_freq_path.exists() and draft_vocab_size is not None:
            import numpy as np
            token_freq_dict = torch.load(token_freq_path, weights_only=True)
            target_vocab_size = get_target_vocab_size(None, verifier)
            d2t, t2d = build_vocab_mappings_from_distribution(
                token_freq_dict=token_freq_dict,
                draft_vocab_size=draft_vocab_size,
                target_vocab_size=target_vocab_size,
            )
            np.save(d2t_path, d2t.cpu().numpy())
            np.save(t2d_path, t2d.cpu().numpy())
        else:
            # Use full verifier vocab
            verifier_config = AutoConfig.from_pretrained(verifier, trust_remote_code=trust_remote_code)
            if hasattr(verifier_config, "text_config"):
                verifier_config = verifier_config.text_config
            draft_vocab_size = verifier_config.vocab_size

        # -- Build transformer layer config for draft model --
        from speculators.models.eagle3.config import Eagle3SpeculatorConfig

        # Import train.py's helper for building the config
        # We replicate the essential logic here to avoid depending on the script's
        # global `args` variable.
        from transformers import PretrainedConfig
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.auto.configuration_auto import AutoConfig as _AutoConfig

        verifier_config = _AutoConfig.from_pretrained(verifier, trust_remote_code=trust_remote_code)
        if hasattr(verifier_config, "text_config"):
            verifier_config = verifier_config.text_config

        hidden_act = (
            getattr(verifier_config, "hidden_act", None)
            or getattr(verifier_config, "hidden_activation", None)
        )

        head_dim = getattr(verifier_config, "head_dim", None)
        num_attention_heads = verifier_config.num_attention_heads
        num_key_value_heads = verifier_config.num_key_value_heads

        if (
            head_dim
            and verifier_config.hidden_size % num_attention_heads != 0
            and verifier_config.hidden_size % head_dim == 0
        ):
            num_attention_heads = verifier_config.hidden_size // head_dim
            if num_attention_heads % num_key_value_heads != 0:
                num_key_value_heads = num_attention_heads

        transformer_layer_config = LlamaConfig(
            vocab_size=verifier_config.vocab_size,
            hidden_size=verifier_config.hidden_size,
            intermediate_size=verifier_config.intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=verifier_config.max_position_embeddings,
            initializer_range=verifier_config.initializer_range,
            rms_norm_eps=verifier_config.rms_norm_eps,
            head_dim=head_dim,
            tie_word_embeddings=False,
        )

        # Copy RoPE config
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) >= version.parse("5.0.0"):
            if hasattr(verifier_config, "rope_parameters"):
                from copy import deepcopy
                transformer_layer_config.rope_parameters = deepcopy(verifier_config.rope_parameters)
        else:
            if hasattr(verifier_config, "rope_scaling"):
                from copy import deepcopy
                transformer_layer_config.rope_scaling = deepcopy(verifier_config.rope_scaling)
            transformer_layer_config.rope_theta = getattr(verifier_config, "rope_theta", 10000.0)

        # -- Create draft model --
        registry = SpeculatorModel.registry
        if not registry or speculator_type not in registry:
            available = list(registry.keys()) if registry else []
            raise ValueError(
                f"Unknown speculator type: '{speculator_type}'. Available: {available}"
            )

        model_class = registry[speculator_type]

        mask_token_id = resolve_mask_token_id(
            verifier,
            transformer_layer_config.vocab_size,
            None,
            trust_remote_code=trust_remote_code,
        )

        if from_pretrained:
            draft_model = model_class.from_pretrained(from_pretrained, t2d=t2d, d2t=d2t)
        else:
            # Build kwargs matching what train.py's main() passes to from_training_args
            model_kwargs = {
                "verifier_name_or_path": verifier,
                "draft_vocab_size": draft_vocab_size,
                "num_layers": num_layers,
                "norm_before_residual": params.get("norm_before_residual", True),
                "norm_before_fc": params.get("norm_before_fc", False),
                "embed_requires_grad": params.get("embed_requires_grad", False),
                "ttt_steps": ttt_steps,
                "target_layer_ids": target_layer_ids,
                "mask_token_id": mask_token_id,
            }
            draft_model = model_class.from_training_args(
                verifier_config=transformer_layer_config,
                t2d=t2d,
                d2t=d2t,
                **model_kwargs,
            )

        # -- Datasets --
        noise_transform = AddUniformNoise(std=noise_std)

        request_timeout = params.get("request_timeout", 300 if on_missing == "generate" else 120)

        train_dataset = ArrowDataset(
            datapath=data_output_dir,
            max_len=total_seq_len,
            hidden_states_path=hidden_states_path,
            vllm_endpoint=vllm_endpoint or "http://localhost:8000/v1",
            on_missing=on_missing,
            transform=noise_transform,
            split_ratio=0.9,
            model=verifier if on_missing == "generate" else None,
            hidden_states_dtype=hidden_states_dtype,
            request_timeout=request_timeout,
        )
        val_dataset = ArrowDataset(
            datapath=data_output_dir,
            max_len=total_seq_len,
            hidden_states_path=hidden_states_path,
            vllm_endpoint=vllm_endpoint or "http://localhost:8000/v1",
            on_missing=on_missing,
            split_ratio=-0.1,
            model=verifier if on_missing == "generate" else None,
            hidden_states_dtype=hidden_states_dtype,
            request_timeout=request_timeout,
        )

        # -- Dataloaders --
        hidden_size = transformer_layer_config.hidden_size
        preprocess = shift_batch if speculator_type in ("eagle3", "peagle") else None

        train_sampler = MultipackDistributedBatchSamplerV2(
            batch_max_length=total_seq_len,
            lengths=train_dataset.approx_lengths,
            num_replicas=1,
            rank=0,
        )
        val_sampler = MultipackDistributedBatchSamplerV2(
            batch_max_length=total_seq_len,
            lengths=val_dataset.approx_lengths,
            num_replicas=1,
            rank=0,
        )

        from torch.utils.data import DataLoader

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            prefetch_factor=4,
            pin_memory=True,
            collate_fn=create_collate_fn(total_seq_len, hidden_size, hidden_states_dtype, preprocess),
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=num_workers,
            prefetch_factor=4,
            pin_memory=True,
            collate_fn=create_collate_fn(total_seq_len, hidden_size, hidden_states_dtype, preprocess),
            persistent_workers=True,
        )

        # -- Trainer kwargs --
        loss_fn = resolve_loss_fn(loss_fn_name)
        train_call_kwargs = {
            "use_off_policy_tokens": params.get("use_off_policy_tokens", False),
            "ttt_steps": ttt_steps,
            "ttt_step_loss_decay": params.get("ttt_step_loss_decay", 1.0),
            "loss_fn": loss_fn,
        }
        val_call_kwargs = {
            "use_off_policy_tokens": False,
            "ttt_steps": ttt_steps,
            "ttt_step_loss_decay": params.get("ttt_step_loss_decay", 1.0),
            "loss_fn": loss_fn,
        }

        # -- GPU selection for training --
        training_gpu_id = params.get("training_gpu_id", 0)
        import torch as _torch
        _torch.cuda.set_device(training_gpu_id)
        logger.info("Training on GPU %d", training_gpu_id)

        # -- Run training --
        trainer_config = TrainerConfig(
            num_epochs=epochs,
            save_path=ckpt_dir,
            lr=lr,
            resume_from_checkpoint=not params.get("no_resume", False),
            is_distributed=False,  # Single GPU by default
            local_rank=training_gpu_id,
            train_call_kwargs=train_call_kwargs,
            val_call_kwargs=val_call_kwargs,
            scheduler_type=scheduler_type,
            checkpoint_freq=checkpoint_freq,
            log_freq=log_freq,
            hidden_states_dtype=hidden_states_dtype,
        )

        trainer = Trainer(draft_model, trainer_config, train_loader, val_loader)
        trainer.run_training()

        # Read validation metrics from best checkpoint
        val_metrics = {}
        best_link = Path(ckpt_dir) / "checkpoint_best"
        if best_link.exists():
            best_target = best_link.resolve()
            val_metrics_file = best_target / "val_metrics.json"
            if val_metrics_file.exists():
                with open(val_metrics_file) as f:
                    val_metrics = json.load(f)

        return {
            "speculator_type": speculator_type,
            "verifier_model": verifier,
            "epochs_completed": epochs,
            "val_metrics": val_metrics,
        }


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class SpeculativeDecodingAlgorithm(Algorithm):
    """Speculative decoding draft model training.

    Trains lightweight draft models (Eagle3, DFlash, MTP, PEagle) to accelerate
    inference of large verifier models through speculative decoding.
    """

    def __init__(self, backend: Backend, **kwargs):
        self.backend = backend
        self.config = kwargs

    def train(
        self,
        verifier_name_or_path: str,
        ckpt_output_dir: str,
        # Data source
        data_path: Optional[str] = None,
        # Mode
        mode: Optional[str] = None,
        # Speculator configuration
        speculator_type: Optional[str] = None,
        draft_vocab_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        target_layer_ids: Optional[List[int]] = None,
        # Training hyperparameters
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        total_seq_len: Optional[int] = None,
        ttt_steps: Optional[int] = None,
        noise_std: Optional[float] = None,
        loss_fn: Optional[str] = None,
        scheduler_type: Optional[str] = None,
        checkpoint_freq: Optional[float] = None,
        log_freq: Optional[int] = None,
        # Hidden state generation
        vllm_endpoint: Optional[str] = None,
        num_gpus: Optional[int] = None,
        hidden_states_path: Optional[str] = None,
        data_output_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        datagen_concurrency: Optional[int] = None,
        # vLLM server config
        vllm_gpu_ids: Optional[List[int]] = None,
        vllm_gpu_memory_utilization: Optional[float] = None,
        vllm_startup_timeout: Optional[int] = None,
        trust_remote_code: Optional[bool] = None,
        # Training GPU
        training_gpu_id: Optional[int] = None,
        # Model config
        norm_before_residual: Optional[bool] = None,
        norm_before_fc: Optional[bool] = None,
        embed_requires_grad: Optional[bool] = None,
        from_pretrained: Optional[str] = None,
        hidden_states_dtype: Optional[str] = None,
        use_off_policy_tokens: Optional[bool] = None,
        ttt_step_loss_decay: Optional[float] = None,
        num_workers: Optional[int] = None,
        no_resume: Optional[bool] = None,
        **kwargs,
    ) -> Any:
        """Train a speculative decoding draft model.

        Args:
            verifier_name_or_path: HF model ID or path to the verifier (base) model.
            ckpt_output_dir: Directory to save trained draft model checkpoints.

            Data Source:
                data_path: Path to sharegpt-format JSON/JSONL, HF dataset ID,
                    or the string "sharegpt" for the built-in dataset.

            Pipeline Mode:
                mode: Pipeline mode (default: "offline"):
                    - "offline": Full pipeline (data prep -> hidden state gen -> training)
                    - "online": Data prep + training with on-demand hidden state gen
                    - "train_only": Train from pre-existing hidden states on disk
                    - "data_only": Generate data and hidden states only

            Speculator Configuration:
                speculator_type: Model architecture (default: "eagle3").
                    Options: eagle3, dflash, mtp, peagle.
                draft_vocab_size: Vocabulary size for draft model. If None, auto-
                    detected from token frequencies or uses full verifier vocab.
                num_layers: Number of draft model decoder layers (default: 1).
                target_layer_ids: Verifier layer IDs to extract hidden states from.
                    Default: auto-computed as [2, n//2, n-3, n].

            Training Hyperparameters:
                epochs: Number of training epochs (default: 3).
                lr: Learning rate (default: 1e-4).
                total_seq_len: Sequence length for training (default: 2048).
                ttt_steps: Test-time training steps (default: 3).
                noise_std: Noise augmentation std (default: 0.05).
                loss_fn: Loss function - "kl_div" or "ce" (default: "kl_div").
                scheduler_type: LR scheduler - "linear", "cosine", "none" (default: "linear").

            Hidden State Generation:
                vllm_endpoint: URL of a running vLLM server. If not provided,
                    a managed server is launched and stopped automatically.
                num_gpus: Number of GPUs for vLLM data-parallel generation (default: 1).
                    Ignored if vllm_gpu_ids is set.
                hidden_states_path: Path to pre-generated hidden states (for train_only).
                data_output_dir: Directory for intermediate data (Arrow dataset, token freqs).
                max_samples: Maximum samples to use from dataset.

            GPU Allocation:
                vllm_gpu_ids: GPU device IDs for managed vLLM server (e.g. [0, 1]).
                    Sets CUDA_VISIBLE_DEVICES on the vLLM subprocess. Also sets
                    num_gpus automatically. If not provided, vLLM uses whatever
                    GPUs are visible.
                training_gpu_id: GPU device ID for training (default: 0).
                    In online mode with managed vLLM, use this together with
                    vllm_gpu_ids to avoid GPU contention.

        Returns:
            Dict with training results including checkpoint_path, timing, and metrics.
        """
        params = {
            "verifier_name_or_path": verifier_name_or_path,
            "ckpt_output_dir": ckpt_output_dir,
        }

        optional_params = {
            "data_path": data_path,
            "mode": mode,
            "speculator_type": speculator_type,
            "draft_vocab_size": draft_vocab_size,
            "num_layers": num_layers,
            "target_layer_ids": target_layer_ids,
            "epochs": epochs,
            "lr": lr,
            "total_seq_len": total_seq_len,
            "ttt_steps": ttt_steps,
            "noise_std": noise_std,
            "loss_fn": loss_fn,
            "scheduler_type": scheduler_type,
            "checkpoint_freq": checkpoint_freq,
            "log_freq": log_freq,
            "vllm_endpoint": vllm_endpoint,
            "num_gpus": num_gpus,
            "hidden_states_path": hidden_states_path,
            "data_output_dir": data_output_dir,
            "max_samples": max_samples,
            "datagen_concurrency": datagen_concurrency,
            "vllm_gpu_ids": vllm_gpu_ids,
            "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
            "vllm_startup_timeout": vllm_startup_timeout,
            "trust_remote_code": trust_remote_code,
            "training_gpu_id": training_gpu_id,
            "norm_before_residual": norm_before_residual,
            "norm_before_fc": norm_before_fc,
            "embed_requires_grad": embed_requires_grad,
            "from_pretrained": from_pretrained,
            "hidden_states_dtype": hidden_states_dtype,
            "use_off_policy_tokens": use_off_policy_tokens,
            "ttt_step_loss_decay": ttt_step_loss_decay,
            "num_workers": num_workers,
            "no_resume": no_resume,
        }

        for key, value in optional_params.items():
            if value is not None:
                params[key] = value

        params.update(kwargs)

        return self.backend.execute_training(params)

    def get_required_params(self) -> Dict[str, Type]:
        return {
            "verifier_name_or_path": str,
            "ckpt_output_dir": str,
        }

    def get_optional_params(self) -> Dict[str, Type]:
        return {
            "data_path": str,
            "mode": str,
            "speculator_type": str,
            "draft_vocab_size": int,
            "num_layers": int,
            "target_layer_ids": list,
            "epochs": int,
            "lr": float,
            "total_seq_len": int,
            "ttt_steps": int,
            "noise_std": float,
            "loss_fn": str,
            "scheduler_type": str,
            "checkpoint_freq": float,
            "log_freq": int,
            "vllm_endpoint": str,
            "num_gpus": int,
            "hidden_states_path": str,
            "data_output_dir": str,
            "max_samples": int,
            "datagen_concurrency": int,
            "vllm_gpu_ids": list,
            "vllm_gpu_memory_utilization": float,
            "vllm_startup_timeout": int,
            "trust_remote_code": bool,
            "training_gpu_id": int,
            "norm_before_residual": bool,
            "norm_before_fc": bool,
            "embed_requires_grad": bool,
            "from_pretrained": str,
            "hidden_states_dtype": str,
            "use_off_policy_tokens": bool,
            "ttt_step_loss_decay": float,
            "num_workers": int,
            "no_resume": bool,
        }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

AlgorithmRegistry.register_algorithm("speculative_decoding", SpeculativeDecodingAlgorithm)
AlgorithmRegistry.register_backend("speculative_decoding", "speculators", SpeculatorsBackend)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def train_speculator(
    verifier_name_or_path: str,
    ckpt_output_dir: str,
    data_path: Optional[str] = None,
    mode: str = "offline",
    speculator_type: str = "eagle3",
    backend: str = "speculators",
    **kwargs,
) -> Any:
    """Train a speculative decoding draft model.

    This is the main entry point for speculative decoding training.
    Wraps the speculators library pipeline.

    Args:
        verifier_name_or_path: HF model ID or path to the base (verifier) model.
        ckpt_output_dir: Directory to save trained draft model checkpoints.
        data_path: Path to sharegpt-format JSON/JSONL, HF dataset ID,
            or "sharegpt" for the built-in dataset.
        mode: Pipeline mode - "offline" (default), "online", "train_only", "data_only".
        speculator_type: Draft model architecture (default: "eagle3").
        backend: Backend to use (default: "speculators").
        **kwargs: Additional parameters passed to the algorithm.

    Returns:
        Dict with training results.

    Examples::

        # Full offline pipeline
        result = train_speculator(
            verifier_name_or_path="Qwen/Qwen3-8B",
            data_path="sharegpt",
            ckpt_output_dir="./eagle3_output",
            epochs=3,
            draft_vocab_size=32000,
        )

        # Train from pre-generated hidden states
        result = train_speculator(
            verifier_name_or_path="Qwen/Qwen3-8B",
            ckpt_output_dir="./eagle3_output",
            mode="train_only",
            data_output_dir="./data",
            hidden_states_path="./data/hidden_states",
        )
    """
    from . import create_algorithm

    algorithm = create_algorithm("speculative_decoding", backend)
    return algorithm.train(
        verifier_name_or_path=verifier_name_or_path,
        ckpt_output_dir=ckpt_output_dir,
        data_path=data_path,
        mode=mode,
        speculator_type=speculator_type,
        **kwargs,
    )


def eagle3(
    verifier_name_or_path: str,
    ckpt_output_dir: str,
    data_path: Optional[str] = None,
    mode: str = "offline",
    **kwargs,
) -> Any:
    """Train an Eagle3 speculative decoding draft model.

    Convenience wrapper for ``train_speculator`` with ``speculator_type="eagle3"``.

    Args:
        verifier_name_or_path: HF model ID or path to the base model.
        ckpt_output_dir: Directory to save checkpoints.
        data_path: Training data path or dataset name.
        mode: Pipeline mode (default: "offline").
        **kwargs: Additional parameters.

    Returns:
        Dict with training results.

    Example::

        from training_hub import eagle3

        result = eagle3(
            "Qwen/Qwen3-8B",
            "./eagle3_output",
            data_path="sharegpt",
            epochs=3,
        )
    """
    return train_speculator(
        verifier_name_or_path=verifier_name_or_path,
        ckpt_output_dir=ckpt_output_dir,
        data_path=data_path,
        mode=mode,
        speculator_type="eagle3",
        **kwargs,
    )


def prepare_speculator_data(
    verifier_name_or_path: str,
    data_path: str,
    data_output_dir: str,
    **kwargs,
) -> Any:
    """Prepare data and generate hidden states for speculator training.

    Runs only data preparation and hidden state generation (no training).
    Equivalent to ``train_speculator(..., mode="data_only")``.

    Args:
        verifier_name_or_path: HF model ID or path to the base model.
        data_path: Training data path or dataset name.
        data_output_dir: Directory to save preprocessed data and hidden states.
        **kwargs: Additional parameters (num_gpus, max_samples, etc.).

    Returns:
        Dict with data_output_dir and hidden_states_path.
    """
    return train_speculator(
        verifier_name_or_path=verifier_name_or_path,
        ckpt_output_dir=data_output_dir,
        data_path=data_path,
        mode="data_only",
        data_output_dir=data_output_dir,
        **kwargs,
    )
