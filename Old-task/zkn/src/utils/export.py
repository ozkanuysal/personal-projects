from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import io
import numpy as np
import torch
import torch.onnx
import json

DEFAULT_ONNX_VERSION = 10
DEFAULT_SCALE = 7
DEFAULT_BATCH_SIZE = 1

class ExportError(Exception):
    """Custom exception for export errors."""
    pass

def validate_inputs(
    input_shape: Optional[List[int]], 
    input_array: Optional[np.ndarray]
) -> None:
    """Validate input parameters."""
    if (input_shape is None) == (input_array is None):
        raise ValueError("Exactly one of input_shape or input_array must be specified")

def prepare_input_tensor(
    input_shape: Optional[List[int]], 
    input_array: Optional[np.ndarray]
) -> torch.Tensor:
    """Prepare input tensor for model export."""
    if input_array is None:
        return 0.1 * torch.rand(1, *input_shape, requires_grad=True)
    
    x = torch.tensor(input_array)
    if input_shape is not None and tuple(input_shape) != x.shape:
        raise ValueError(f"Input shape {input_shape} doesn't match array shape {x.shape}")
    
    new_shape = tuple([1] + list(x.shape))
    return torch.reshape(x, new_shape)

def export_to_onnx(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    onnx_filename: Union[str, Path]
) -> None:
    """Export model to ONNX format."""
    try:
        torch.onnx.export(
            model,
            input_tensor,
            onnx_filename,
            export_params=True,
            opset_version=DEFAULT_ONNX_VERSION,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
    except Exception as e:
        raise ExportError(f"Failed to export model to ONNX: {str(e)}")

def prepare_input_output_data(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor
) -> Dict[str, List[List[float]]]:
    """Prepare input/output data for JSON export."""
    return {
        "input_data": [input_tensor.detach().numpy().reshape([-1]).tolist()],
        "output_data": [o.detach().numpy().reshape([-1]).tolist() for o in output_tensor]
    }

def export(
    torch_model: torch.nn.Module,
    input_shape: Optional[List[int]] = None,
    input_array: Optional[np.ndarray] = None,
    onnx_filename: Union[str, Path] = "network.onnx",
    input_filename: Union[str, Path] = "input.json",
) -> None:
    """
    Export a PyTorch model to ONNX format with input/output data.
    
    Args:
        torch_model: PyTorch model to export
        input_shape: Shape of random input to generate
        input_array: Specific input array to use
        onnx_filename: Output ONNX filename
        input_filename: Output JSON filename for input/output data
    
    Raises:
        ValueError: If input parameters are invalid
        ExportError: If export fails
    """
    validate_inputs(input_shape, input_array)
    input_tensor = prepare_input_tensor(input_shape, input_array)
    
    # Set model to eval mode if possible
    if hasattr(torch_model, 'eval'):
        torch_model.eval()
    
    # Get model output
    try:
        output_tensor = torch_model(input_tensor)
    except Exception as e:
        raise ExportError(f"Failed to run model inference: {str(e)}")
    
    # Export model to ONNX
    export_to_onnx(torch_model, input_tensor, onnx_filename)
    
    # Save input/output data
    data = prepare_input_output_data(input_tensor, output_tensor)
    try:
        with open(input_filename, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        raise ExportError(f"Failed to save input/output data: {str(e)}")