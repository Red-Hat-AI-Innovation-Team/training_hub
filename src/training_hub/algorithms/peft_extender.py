"""
PEFT (Parameter-Efficient Fine-Tuning) extender for training hub algorithms.

This module provides common PEFT functionality that can be combined with any base algorithm
(SFT, OSFT, etc.) to enable parameter-efficient training techniques like LoRA, AdaLoRA, etc.
"""

from typing import Any, Dict, List, Optional, Type, Union
from abc import ABC, abstractmethod


class PEFTExtender(ABC):
    """Base class for PEFT extensions that can be combined with any algorithm."""

    @abstractmethod
    def get_peft_params(self) -> Dict[str, Type]:
        """Return PEFT-specific parameters and their types."""
        pass

    @abstractmethod
    def apply_peft_config(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply PEFT configuration to base algorithm parameters."""
        pass


class LoRAPEFTExtender(PEFTExtender):
    """LoRA-specific PEFT extender that provides LoRA parameters and configuration."""

    def get_peft_params(self) -> Dict[str, Type]:
        """Return LoRA-specific parameters and their types."""
        return {
            # Core LoRA parameters
            'lora_r': int,
            'lora_alpha': int,
            'lora_dropout': float,
            'target_modules': List[str],

            # Quantization parameters (QLoRA)
            'load_in_4bit': bool,
            'load_in_8bit': bool,
            'bnb_4bit_quant_type': str,
            'bnb_4bit_compute_dtype': str,
            'bnb_4bit_use_double_quant': bool,

            # Advanced LoRA options
            'use_rslora': bool,
            'use_dora': bool,
            'loftq_config': Dict[str, Any],
            'init_lora_weights': Union[bool, str],
            'rank_pattern': Dict[str, int],
            'alpha_pattern': Dict[str, int],
        }

    def apply_peft_config(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply LoRA configuration to base algorithm parameters."""
        # Extract LoRA-specific parameters
        lora_params = {}
        peft_param_names = set(self.get_peft_params().keys())

        for key, value in base_params.items():
            if key in peft_param_names and value is not None:
                lora_params[key] = value

        # Set default LoRA values if not specified
        lora_defaults = {
            'lora_r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.0,  # Unsloth optimized for 0 dropout
            'target_modules': None,  # Will be auto-detected by backends
            'load_in_4bit': False,
            'load_in_8bit': False,
            'use_rslora': False,
            'use_dora': False,
            'init_lora_weights': True,
        }

        # Apply defaults for missing LoRA parameters
        for key, default_value in lora_defaults.items():
            if key not in lora_params:
                lora_params[key] = default_value

        # Combine with base parameters
        combined_params = base_params.copy()
        combined_params.update(lora_params)

        # Add metadata to indicate PEFT is enabled
        combined_params['peft_enabled'] = True
        combined_params['peft_type'] = 'lora'

        return combined_params


def get_lora_parameters() -> Dict[str, Type]:
    """Convenience function to get LoRA parameter definitions."""
    extender = LoRAPEFTExtender()
    return extender.get_peft_params()


def apply_lora_defaults(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to apply LoRA defaults to parameters."""
    extender = LoRAPEFTExtender()
    return extender.apply_peft_config(params)


# Export commonly used functions and classes
__all__ = [
    'PEFTExtender',
    'LoRAPEFTExtender',
    'get_lora_parameters',
    'apply_lora_defaults'
]