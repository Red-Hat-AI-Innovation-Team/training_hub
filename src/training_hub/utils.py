import os
from typing import Literal, get_origin, get_args
import torch
import warnings

def format_type_name(tp):
    # Handle None
    if tp is type(None):
        return 'None'
    
    # Handle basic types
    if hasattr(tp, '__name__'):
        return tp.__name__
    
    # Handle typing generics
    origin = get_origin(tp)
    args = get_args(tp)
    
    if origin is not None:
        origin_name = getattr(origin, '__name__', str(origin))
        if args:
            arg_names = [format_type_name(arg) for arg in args]
            return f"{origin_name}[{', '.join(arg_names)}]"
        return origin_name
    
    # Fallback: clean up the string representation
    type_str = str(tp)
    if type_str.startswith("<class '") and type_str.endswith("'>"):
        return type_str[8:-2]
    
    return type_str


def get_torchrun_params(args: dict) -> dict[str, str | int]:
    """
    Parse and load PyTorch distributed training parameters with hierarchical precedence.
    
    Precedence order: args dict > environment variables > defaults
    
    When both rdzv_endpoint and master_addr are set at different precedence levels,
    the higher precedence value is used with a warning. If both are set at the same
    level (both in args or both in env), an error is raised.
    
    Args:
        args (dict): Dictionary containing torchrun configuration parameters
    
    Returns:
        dict: Dictionary with validated torchrun parameters
    
    Raises:
        ValueError: If nproc_per_node='auto' but no CUDA GPUs available
        ValueError: If master_addr and rdzv_endpoint are both set at the same precedence level
        ValueError: If nproc_per_node has invalid value
    """
    torchrun_args = {}
    
    def get_env_value(param_name):
        """Get environment variable value with fallback logic."""
        if param_name in ['master_addr', 'master_port']:
            # try both PET_ and non-PET_ versions
            return os.getenv(f'PET_{param_name.upper()}') or os.getenv(param_name.upper())
        return os.getenv(f'PET_{param_name.upper()}')
    
    def validate_env_conflict(param_name):
        """Check if both PET_ and standard env vars are set with conflicting values.
        
        Should only be called when we're actually going to use the env value.
        """
        if param_name in ['master_addr', 'master_port']:
            pet_val = os.getenv(f'PET_{param_name.upper()}')
            standard_val = os.getenv(param_name.upper())
            
            if pet_val and standard_val and pet_val != standard_val:
                raise ValueError(
                    f"Conflicting environment variables: PET_{param_name.upper()}={pet_val!r} "
                    f"and {param_name.upper()}={standard_val!r}. These must match if both are set."
                )
    
    def get_param_value(param_name: str) -> tuple[str | int | None, Literal['args', 'env', None]]:
        """
        Get parameter value following precedence: args > env > None.
        
        The only valid types which torchrun accepts are str and int, so this
        function will at most return str, int, or None.
        
        Returns tuple of (value, source) where source is 'args', 'env', or None.
        
        Args:
            param_name: Name of parameter to retrieve
        """
        # check args dict first - if explicitly set in args, always use it regardless of value
        if param_name in args:
            return args[param_name], 'args'
        
        # check environment variables - here we apply the is_string logic
        env_val = get_env_value(param_name)
        if env_val:
            return env_val, 'env'
        return None, None
 
    def validate_nproc_per_node(value: int | str) -> int | str:
        """Validate and normalize nproc_per_node."""
        if not isinstance(value, (int, str)):
            raise ValueError(f"nproc_per_node must be 'auto', 'gpu', or an integer, got type {type(value).__name__}")
        if isinstance(value, int):
            return value

        value_lower = value.lower().strip()
        if value_lower not in ['auto', 'gpu'] and not value_lower.isdigit():
            raise ValueError(f"nproc_per_node must be 'auto', 'gpu', or an integer, got: {value!r}")
        if value_lower.isdigit():
            return int(value_lower)

        # handle 'auto' and 'gpu' - both require CUDA
        if value_lower in ['auto', 'gpu'] and torch.cuda.is_available():
            return 'gpu'
        else:
            raise ValueError(f"nproc_per_node='{value_lower}' requires CUDA GPUs, but none are available")

    def get_param_reference(param_name: str, source: str) -> str:
        """Format parameter reference based on source (args vs env)."""
        if source == 'args':
            return f"'{param_name}' (from args)"
        elif source == 'env':
            # show the actual env var name
            if param_name in ['master_addr', 'master_port']:
                pet_var = f'PET_{param_name.upper()}'
                std_var = param_name.upper()
                # check which one is actually set
                if os.getenv(pet_var):
                    return f"{pet_var}"
                else:
                    return f"{std_var}"
            else:
                return f"PET_{param_name.upper()}"
        return param_name

    # process nproc_per_node with validation
    nproc_val, _ = get_param_value('nproc_per_node')
    torchrun_args['nproc_per_node'] = validate_nproc_per_node(nproc_val) if nproc_val is not None else 1

    # simple int-only params
    int_params_with_defaults = {
        'nnodes': 1,
        'node_rank': 0,
    }
    for param, default in int_params_with_defaults.items():
        # we know the final values in this case must be integers, so any non-None value here
        # should be castable to `int`.
        value, _ = get_param_value(param)
        if value is not None:
            try:
                torchrun_args[param] = int(value)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid value for {param}: {value!r}. Must be an integer.") from e
        else:
            torchrun_args[param] = default
    
    
    # rdzv_id will be either a str or int; we just perform some cleanup before
    # setting it in the final args dict.
    rdzv_id_val, _ = get_param_value('rdzv_id')
    if isinstance(rdzv_id_val, str):
        rdzv_id_val = rdzv_id_val.strip()
    torchrun_args['rdzv_id'] = rdzv_id_val if rdzv_id_val is not None else 0
 
    
    # process mutually exclusive string parameters with precedence handling
    master_addr_val, master_addr_source = get_param_value('master_addr')
    rdzv_endpoint_val, rdzv_endpoint_source = get_param_value('rdzv_endpoint')

    # Here, when both of these are set, we basically want to either error out or 
    # set the one with lower precedence to None
    if master_addr_val and rdzv_endpoint_val:
        master_addr_ref = get_param_reference('master_addr', master_addr_source)
        rdzv_endpoint_ref = get_param_reference('rdzv_endpoint', rdzv_endpoint_source)
        
        # both are set - check if they're from the same precedence level
        if master_addr_source == rdzv_endpoint_source:
            raise ValueError(
                f"Cannot specify both {master_addr_ref}={master_addr_val!r} and "
                f"{rdzv_endpoint_ref}={rdzv_endpoint_val!r} at the same level ({master_addr_source}). "
                "These parameters are mutually exclusive."
            )
        # different precedence levels - use the higher precedence one with a warning
        if master_addr_source == 'args':
            warnings.warn(
                f"Both {master_addr_ref}={master_addr_val!r} and {rdzv_endpoint_ref}={rdzv_endpoint_val!r} are set. "
                f"Using {master_addr_ref} due to higher precedence. Ignoring {rdzv_endpoint_ref}.",
                UserWarning
            )
            rdzv_endpoint_val = None

        else:  # rdzv_endpoint_source == 'args'
            warnings.warn(
                f"Both {rdzv_endpoint_ref}={rdzv_endpoint_val!r} and {master_addr_ref}={master_addr_val!r} are set. "
                f"Using {rdzv_endpoint_ref} due to higher precedence. Ignoring {master_addr_ref}.",
                UserWarning
            )
            master_addr_val = None

    # no conflict, add whichever is set
    # It may also be possible for neither to be set, in which case we want the 
    # library/torchrun to figure out what to do
    if master_addr_val:
        # validate env conflicts only when we're actually using master_addr
        if master_addr_source == 'env':
            validate_env_conflict('master_addr')
        
        # we only want to set port when master addr is also being set, to avoid
        # any potential confusion
        torchrun_args['master_addr'] = master_addr_val
        master_port_val, master_port_source = get_param_value('master_port')
        if master_port_val is not None:
            # validate env conflicts only when we're actually using master_port
            if master_port_source == 'env':
                validate_env_conflict('master_port')
            try:
                torchrun_args['master_port'] = int(master_port_val)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid value for master_port: {master_port_val!r}. Must be an integer.") from e

    # Note: If neither master_addr nor rdzv_endpoint is set, torchrun will use
    # its default behavior (typically localhost or other configured defaults)
    elif rdzv_endpoint_val:
        torchrun_args['rdzv_endpoint'] = rdzv_endpoint_val

    return torchrun_args
