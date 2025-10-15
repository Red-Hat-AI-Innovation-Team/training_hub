from typing import override
from transformers import AutoModel
from mini_trainer.osft_utils import MODEL_CONFIGS
import numpy as np

"""
Code assisted by Cursor/Claude4
"""

# Constants that specify
FLOAT32_BYTES_N: int = 4
FLOAT16_BYTES_N: int = 2
FLOAT8_BYTES_N: int = 1
ADAMW_PARAMS_N: int = 2

# Helper function to do the rounding when printing 
def ROUNDER(value: int) -> str: return str(round(value / 1073741824, 1))

# Helper function to calculate how much the given unfrozen_rank_ratio 
# will be affecting the OSFT estimation (through a quadratic mapping where
# 0 is 0.5 of SFT's value, 1/3 is equal to SFT's value, and 1 is twice of SFT's value)
def OSFT_RATIO(value: float) -> float: return 0.5 + (1.5 * value)


class BasicEstimator:
    """
    An estimator for the memory usage of a typical LLM training pass via SFT.
    Estimators for other methods can subclass this class.

    Args:
        num_gpus (int): Number of GPUs to use for training (default: 8)
        gpu_memory (int): The VRAM of each GPU in bytes (default: 85899345920 for 80 GB)
        model_path (str): HuggingFace model path to the model to fine-tune
                        (default: "ibm-granite/granite-3.3-8b-instruct")
        effective_batch_size (int): The number of samples in a minibatch that the model
                                    has to see before backpropping.
        max_seq_len (int): Maximum sequence length of dataset samples 
        max_tokens_per_gpu (int): The maximum number of tokens that can be placed
                                on a single GPU during each mini-batch.
        use_liger (bool): If true, estimate assuming Liger Kernels are used.
        verbose (int): The level of verbosity to print out. Set to 0 for no printing,
                        set to 1 to print out only hardware recommendations,
                        set to 2 for a detailed memory breakdown.
    """

    def __init__(
        self,
        num_gpus: int = 8,
        gpu_memory: int = 85899345920,
        model_path: str = "ibm-granite/granite-3.3-8b-instruct",
        effective_batch_size: int | None = None,
        max_seq_len: int | None = None,
        max_tokens_per_gpu: int | None = None,
        use_liger: bool = False,
        verbose: int = 1,
        trust_remote_code: bool = False,
    ):
        self.num_gpus = num_gpus
        self.gpu_memory = gpu_memory
        self.model_path = model_path
        self.use_liger = use_liger
        self.verbose = verbose
        
        # Load model directly
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=trust_remote_code)

        # Determine parameters needed for calculations
        self.num_params: int = self.model.num_parameters(only_trainable=False)
        self.num_trainable_params: int = self.model.num_parameters(only_trainable=True)
        self.num_layers: int = self.model.config.num_hidden_layers
        self.hidden_size: int = self.model.config.hidden_size  

        # The number of tokens on each GPU will be bounded by either
        # The largest number of tokens in a batch (divided by # GPUs)
        # *or* the value of max_tokens_per_gpu
        self.tokens_per_gpu = None            
        if effective_batch_size is None or max_seq_len is None:
            self.tokens_per_gpu: int = max_tokens_per_gpu
        elif max_tokens_per_gpu is None:
            self.tokens_per_gpu: int = effective_batch_size * max_seq_len / self.num_gpus
        else:
            self.tokens_per_gpu: int = min(max_tokens_per_gpu,
                                            effective_batch_size * max_seq_len / self.num_gpus)

        if self.tokens_per_gpu is None:
            raise ValueError("At least one of (effective_batch_size, max_seq_len) or " +
                                "max_tokens_per_gpu must be provided")

        # This is a scalar that's applied during the output memory calculation
        self.output_constant = 8/3
        self.main_dtype_bytes = FLOAT32_BYTES_N
        self.opt_params = ADAMW_PARAMS_N


    def _calc_model_params(self):
        """
        Calcuate the VRAM for storing the model parameters
        """
        return self.num_params * self.main_dtype_bytes / self.num_gpus


    def _calc_gradients(self):
        """
        Calcuate the VRAM for storing the gradients
        """
        return (self.main_dtype_bytes * self.num_trainable_params / self.num_gpus)


    def _calc_optimizer(self):
        """
        Calculate the VRAM for storing the optimizer states
        """
        return (self.main_dtype_bytes * self.num_trainable_params * self.opt_params / self.num_gpus)


    def _calc_intermediate_activations(self):
        """
        Calculate the VRAM for storing the model's intermediate activations
        """
        return (self.tokens_per_gpu * self.main_dtype_bytes  * self.num_layers * self.hidden_size)


    def _calc_outputs(self):
        """
        Calculate the VRAM for storing the model's activated outputs.
        Note that this value is 0 if Liger Kernels are used.
        """
        if not self.use_liger:
            # This nested try-catch attempts to find the model's vocabulary size
            try:
                vocab_size = self.model.embed_tokens.num_embeddings
            except AttributeError:
                try:
                    vocab_size = self.model.config.vocab_size
                except AttributeError:
                    try:
                        vocab_size = self.model.get_input_embeddings().num_embeddings
                    except AttributeError as e:
                        raise ValueError("Could not find the given model's vocabulary size") from e
            return (self.tokens_per_gpu * self.main_dtype_bytes * vocab_size) * self.output_constant
        else:
            return 0


    def _calc_additional(self, **kwargs):
        """
        Calculate any additional VRAM that this training method might need
        """
        return 0


    def _apply_overhead(self, subtotal):
        """
        Apply 0-30% overhead to the subtotal of VRAM needed

        Args:
            subtotal (int): The subtotal of VRAM needed

        Returns:
            gpu_tuple: A triplet of the lower bound, the specific estimate,
                        and upper bound of VRAM required to train this model
            overhead_tuple: A triplet of the lowest, expected, and highest amount of
                            overhead memory required for training
        """
        gpu_vram_total_low: int = int(subtotal)
        gpu_vram_total_mid: int = int(1.1 *  subtotal)
        gpu_vram_total_high: int = int(1.3 * subtotal)
        extra_low: int = max(gpu_vram_total_low - subtotal, 0)
        extra_mid: int = gpu_vram_total_mid - subtotal
        extra_high: int = gpu_vram_total_high - subtotal
        return(gpu_vram_total_low, gpu_vram_total_mid, gpu_vram_total_high), \
                (extra_low, extra_mid, extra_high)


    def _print_results(self, results, overhead, gpu_vram_par, gpu_vram_opt,
                        gpu_vram_grad, gpu_vram_act, gpu_vram_outputs, gpu_vram_additional):
        """
        Print out a breakdown of the estimated memory requirements
        """

        print("Estimations for " + self.model_path + ":\n\n")

        print("Summary:")
        print("The expected amount of memory needed to run this model is about " + 
                ROUNDER(results[1] * self.num_gpus) + " GB") 
        print("The lower and upper bounds are " + 
                ROUNDER(results[0] * self.num_gpus)  + " - " + 
                ROUNDER(results[2] * self.num_gpus) + " GB") 
        print("If you have " + str(self.num_gpus) + " GPUs, you will need about " + \
                ROUNDER(results[1]) + " GB, with bounds of " +
                ROUNDER(results[0]) + " - " + ROUNDER(results[2]) + " GB per GPU")
        print("\n")

        print("Component Breakdown:")
        print("Each GPU will need " + ROUNDER(gpu_vram_par) + " GB to store the model parameters")
        print("Each GPU will need " + ROUNDER(gpu_vram_opt) + " GB to store the optimizer states")
        print("Each GPU will need " + ROUNDER(gpu_vram_grad) + " GB to store the gradients")
        print("Each GPU will need " + ROUNDER(gpu_vram_act) + " GB to store the intermediate activations")
        if self.use_liger:
            print("Since Liger Kernels are being used, no additional memory is needed to store the outputs")
        else:
            print("Each GPU will need " + ROUNDER(gpu_vram_outputs) + " GB to store the outputs")
        if gpu_vram_additional > 0:
            print("This method also requires each GPU to use an additional " + ROUNDER(gpu_vram_additional) + " GB")
        print("Up to " + ROUNDER(overhead[2]) + " GB can be expected as overhead")


    def _print_tips(self, results):
        """
        Print out information on how feasible it is to train the model on
        the user's given hardware and, if necessary, recommend the necessary
        amount of additional memory. 
        """

        min_message = "Minimum extra memory required (to reach the\nlow bound of memory requirements): " + \
            ROUNDER(results[0] - self.gpu_memory) + " GB"
        mid_message = "Recommended amount of extra memory (to reach the\nlikely estimated memory requirements): " + \
            ROUNDER(results[1] - self.gpu_memory) + " GB"
        max_message = "Ideal amount of extra memory required (to reach the\nupper bound of memory requirements): " + \
            ROUNDER(results[2] - self.gpu_memory) + " GB"

        print("\nDecision:")
        
        if results[2] <= self.gpu_memory:
            print("The proposed training setup should work for your hardware.\n")
        elif results[1] <= self.gpu_memory: 
            print("The proposed training setup will likely work for your hardware.\n")
            print(max_message)
        elif results[0] <= self.gpu_memory:
            print("The proposed training setup may work but isn't recommended for your hardware.\n")
            print(mid_message)
            print(max_message)
        else: 
            print("The proposed training setup is impossible for your hardware.\n\n")
            print(min_message)
            print(mid_message)
            print(max_message)


    def estimate(self) -> tuple[int, int, int]:
        """
        Calculate the memory needed to fine tune the given model for the 
        given hyperparameters. After that, determine how possible it is for
        the given hardware to run the model, and note how much more memory
        is needed to make the training more feasible. 

        Note that this estimate assumes training_hub will be used, 
        in which all data types are float32 and the optimizer is always AdamW.

        Return:
            A tuple containing three values:
                lower_bound (int): The lower bound of the memory usage (in bytes)
                expected (int): The expected amount of memory usage (in bytes)
                upper_bound (int): The upper bound of the memory usage (in bytes)
        """  

        # Calculate each piece of memory to be factored into the estimation
        gpu_vram_par = self._calc_model_params()
        gpu_vram_opt: int = self._calc_optimizer()
        gpu_vram_grad: int = self._calc_gradients()
        gpu_vram_act: int = self._calc_intermediate_activations()
        gpu_vram_outputs: int = self._calc_outputs()
        gpu_vram_additional: int = self._calc_additional()

        # Sum up each proposed amount of memory
        subtotal: int = gpu_vram_par + \
                        gpu_vram_opt + \
                        gpu_vram_grad + \
                        gpu_vram_act + \
                        gpu_vram_outputs + \
                        gpu_vram_additional
    
        # Apply some proportion of overhead to get the final memory calculations
        results, overhead = self._apply_overhead(subtotal)

        # Print out details of the memory breakdown and recommendations based
        # on the user's given hardware. 
        if self.verbose > 1:
            self._print_results(results, overhead, gpu_vram_par,
                                gpu_vram_opt, gpu_vram_grad, gpu_vram_act,
                                gpu_vram_outputs, gpu_vram_additional)
        if self.verbose > 0:
            self._print_tips(results)

        # Return the lower bound, estimated value, and upper bound 
        return results 


class OSFTEstimatorExperimental(BasicEstimator):
    """
    An estimator for the memory usage of an LLM trained via OSFT. 
    Subclasses the BasicEstimator class.

    NOTE: This is an experimental implementation of creating a more accurate
    memory estimator for OSFT. However, it is still under development.

    Args (in addition to the BasicEstimator args):
        unfreeze_rank_ratio (float): The portion of the weight matrix that is unfrozen
                                    during OSFT.
    """

    @override
    def __init__(
        self,
        num_gpus: int = 8,
        gpu_memory: int = 85899345920,
        model_path: str = "ibm-granite/granite-3.3-8b-instruct",
        effective_batch_size: int | None = None,
        max_seq_len: int | None = None,
        max_tokens_per_gpu: int | None = None,
        use_liger: bool = False,
        verbose: int = 1,
        trust_remote_code: bool = False,
        unfreeze_rank_ratio: float = 0.25,
    ):
        super().__init__(num_gpus, gpu_memory, model_path,
                        effective_batch_size, max_seq_len, max_tokens_per_gpu, 
                        use_liger, verbose, trust_remote_code)
        self.output_constant = 7/3
        self.unfreeze_rank_ratio = unfreeze_rank_ratio
        if not (0.0 <= self.unfreeze_rank_ratio <= 1.0):
            raise ValueError("Ratio must be in the range [0, 1]")

        # Check to see which terms need to be included in the search for valid layers
        self.target_terms = MODEL_CONFIGS['default']['patterns']
        for key in MODEL_CONFIGS.keys():
            if self.model_path.find(key) > -1:
                self.target_terms = MODEL_CONFIGS[key]['patterns']


    def _check_layer_params(
                        self,
                        layer_name: str
                    ) -> int:
        """
        For the given layer, determine how many bytes would be needed
        to store this layer when trained with OSFT
        """

        # If the layer is trainable, i.e. it contains any of these terms,
        # then we will need to store a SVD decomposition of the layer in memory.
        layer_data = self.model.state_dict()[layer_name]
        if layer_data.dim() < 2:
            return layer_data.numel() * FLOAT32_BYTES_N
        for term in self.target_terms:
            lowest_dim = layer_data.shape[0] if layer_data.shape[0] < \
                            layer_data.shape[1] else layer_data.shape[1]
            if layer_name.find(term) > -1 and layer_name.find('weight') > -1:
                U_bytes_n: int = layer_data.shape[0] * lowest_dim
                S_bytes_n: int = lowest_dim
                V_bytes_n: int = lowest_dim * layer_data.shape[1]
                return (U_bytes_n + S_bytes_n + V_bytes_n) * FLOAT32_BYTES_N

        # If not, we'll only be storing the layer itself in memory. 
        byte_val = layer_data.numel() * FLOAT32_BYTES_N
        return byte_val


    def _calc_osft_params(self) -> int:
        """
        Iterate through the layers in this model and determine how
        many bytes would be needed to store this model when trained with OSFT
        """
        total_bytes: int = 0
        for layer_name in self.model.state_dict().keys():
            total_bytes += self._check_layer_params(layer_name)
        return total_bytes


    @override
    def _calc_model_params(self):
        """
        Override the model parameter calculation by calculating based on OSFT's parameters
        """
        return self._calc_osft_params() / self.num_gpus 

    @override
    def _calc_gradients(self):
        """
        Override the optimizer parameter calculation by calculating based on OSFT's parameters
        """
        return self._calc_model_params() * OSFT_RATIO(self.unfreeze_rank_ratio)

    @override
    def estimate(self) -> tuple[int, int, int]:
        print("CAUTION: This estimator for OSFT's memory requirements is still under development.\n" +
                "Actual memory requirements may vary from the given estimate.")

        return super().estimate()


class OSFTEstimator(BasicEstimator):
    """
    An estimator for the memory usage of an LLM trained via OSFT. 
    Subclasses the BasicEstimator class.

    NOTE: This is a more basic implementation of an OSFT estimator by
    extrapolating from the SFT estimator.
    Please be warned that the estimates may not be fully accurate.

    Args (in addition to the BasicEstimator args):
        unfreeze_rank_ratio (float): The portion of the weight matrix that is unfrozen
                                    during OSFT.
    """

    @override
    def __init__(
        self,
        num_gpus: int = 8,
        gpu_memory: int = 85899345920,
        model_path: str = "ibm-granite/granite-3.3-8b-instruct",
        effective_batch_size: int | None = None,
        max_seq_len: int | None = None,
        max_tokens_per_gpu: int | None = None,
        use_liger: bool = False,
        verbose: int = 1,
        trust_remote_code: bool = False,
        unfreeze_rank_ratio: float = 0.25,
    ):
        super().__init__(num_gpus, gpu_memory, model_path,
                            effective_batch_size, max_seq_len, max_tokens_per_gpu, 
                            use_liger, verbose, trust_remote_code)
        self.unfreeze_rank_ratio = unfreeze_rank_ratio
        if not (0.0 <= self.unfreeze_rank_ratio <= 1.0):
            raise ValueError("Ratio must be in the range [0, 1]")

    @override
    def _apply_overhead(self, subtotal):
        """
        In addition to the 0-30% overhead, apply a multiplier based on the unfreeze_rank_ratio
        """
        ratio_val = OSFT_RATIO(self.unfreeze_rank_ratio)
        return super()._apply_overhead(subtotal * ratio_val)        
    
    @override
    def estimate(self) -> tuple[int, int, int]:
        print("CAUTION: This is a very rough estimate of OSFT's memory requirements.\n" +
                "Actual memory requirements may vary from the given estimate.")

        return super().estimate()


def estimate(
        training_method: str = "sft",
        num_gpus: int = 8,
        gpu_memory: int = 85899345920,
        model_path: str = "ibm-granite/granite-3.3-8b-instruct",
        effective_batch_size: int | None = None,
        max_seq_len: int | None = None,
        max_tokens_per_gpu: int | None = None,
        use_liger: bool = False,
        verbose: int = 1,
        trust_remote_code: bool = False,
        unfreeze_rank_ratio: float = 0.25,
    ):
    """
    Convenience function for performing estimation

    Args:
        training_method (str): The training method to estimate the memory for. 
                                Set to "osft" to estimate for OSFT,
                                By default, SFT is assumed.
        num_gpus (int): Number of GPUs to use for training (default: 8)
        gpu_memory (int): The VRAM of each GPU in bytes (default: 85899345920 for 80 GB)
        model_path (str): HuggingFace model path to the model to fine-tune
                        (default: "ibm-granite/granite-3.3-8b-instruct")
        effective_batch_size (int): The number of samples in a minibatch that the model
                                    has to see before backpropping.
        max_seq_len (int): Maximum sequence length of dataset samples 
        max_tokens_per_gpu (int): The maximum number of tokens that can be placed
                                on a single GPU during each mini-batch.
        use_liger (bool): If true, estimate assuming Liger Kernels are used.
        verbose (int): The level of verbosity to print out. Set to 0 for no printing,
                        set to 1 to print out only hardware recommendations,
                        set to 2 for a detailed memory breakdown.
        unfreeze_rank_ratio (float): The portion of the weight matrix that is unfrozen
                                    during OSFT.

    Return:
        A tuple containing three values:
            lower_bound (int): The lower bound of the memory usage (in bytes)
            expected (int): The expected amount of memory usage (in bytes)
            upper_bound (int): The upper bound of the memory usage (in bytes)
    """
    
    if training_method.lower() == "osft":
        estimator = OSFTEstimator(num_gpus,
                                    gpu_memory,
                                    model_path,
                                    effective_batch_size,
                                    max_seq_len,
                                    max_tokens_per_gpu,
                                    use_liger,
                                    verbose,
                                    trust_remote_code,
                                    unfreeze_rank_ratio,
                                )
    elif training_method.lower() == "osft-e":
        estimator = OSFTEstimatorExperimental(num_gpus,
                                    gpu_memory,
                                    model_path,
                                    effective_batch_size,
                                    max_seq_len,
                                    max_tokens_per_gpu,
                                    use_liger,
                                    verbose,
                                    trust_remote_code,
                                    unfreeze_rank_ratio,
                                )
    else:
        estimator = BasicEstimator(num_gpus,
                                    gpu_memory,
                                    model_path,
                                    effective_batch_size,
                                    max_seq_len,
                                    max_tokens_per_gpu,
                                    use_liger,
                                    verbose, 
                                    trust_remote_code
                                )
    
    return estimator.estimate()
