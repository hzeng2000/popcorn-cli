import torch
from typing import Callable, Any


def make_match_reference(
    ref_kernel: Callable[[Any], torch.Tensor],
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> Callable[[Callable[[Any], torch.Tensor], Any], bool]:
    """
    Create a function that checks if an implementation matches the reference.
    
    Args:
        ref_kernel: The reference kernel function
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
    
    Returns:
        A function that takes (impl_kernel, input_data) and returns True if outputs match
    """
    def check_implementation(impl_kernel: Callable[[Any], torch.Tensor], data: Any) -> bool:
        ref_output = ref_kernel(data)
        impl_output = impl_kernel(data)
        
        if not isinstance(impl_output, torch.Tensor):
            print(f"Error: Implementation output is not a tensor, got {type(impl_output)}")
            return False
        
        if ref_output.shape != impl_output.shape:
            print(f"Error: Shape mismatch - reference: {ref_output.shape}, implementation: {impl_output.shape}")
            return False
        
        if ref_output.dtype != impl_output.dtype:
            print(f"Warning: Dtype mismatch - reference: {ref_output.dtype}, implementation: {impl_output.dtype}")
        
        if not torch.allclose(ref_output, impl_output, rtol=rtol, atol=atol):
            max_diff = (ref_output - impl_output).abs().max().item()
            print(f"Error: Output mismatch - max difference: {max_diff}")
            return False
        
        return True
    
    return check_implementation
