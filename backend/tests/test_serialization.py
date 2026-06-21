import json

import numpy as np
import torch
from utils.serialization import sanitize_numpy


def test_sanitize_numpy_scalars():
    # Test numpy float
    np_float = np.float64(3.14159)
    sanitized_float = sanitize_numpy(np_float)
    assert isinstance(sanitized_float, float)
    assert sanitized_float == 3.14159

    # Test numpy integer
    np_int = np.int64(42)
    sanitized_int = sanitize_numpy(np_int)
    assert isinstance(sanitized_int, int)
    assert sanitized_int == 42

    # Test numpy boolean
    np_bool = np.bool_(True)
    sanitized_bool = sanitize_numpy(np_bool)
    assert isinstance(sanitized_bool, bool)
    assert sanitized_bool is True

def test_sanitize_numpy_arrays():
    np_arr = np.array([1, 2, 3], dtype=np.int32)
    sanitized_arr = sanitize_numpy(np_arr)
    assert isinstance(sanitized_arr, list)
    assert all(isinstance(x, int) for x in sanitized_arr)
    assert sanitized_arr == [1, 2, 3]

def test_sanitize_torch_tensors():
    tensor_scalar = torch.tensor(2.718)
    sanitized_scalar = sanitize_numpy(tensor_scalar)
    assert isinstance(sanitized_scalar, float)
    assert round(sanitized_scalar, 3) == 2.718

    tensor_arr = torch.tensor([4, 5, 6])
    sanitized_arr = sanitize_numpy(tensor_arr)
    assert isinstance(sanitized_arr, list)
    assert sanitized_arr == [4, 5, 6]

def test_sanitize_nested_structures():
    nested_data = {
        "status": "success",
        "scores": np.array([0.9, 0.1, 0.5], dtype=np.float32),
        "metrics": {
            "accuracy": np.float64(0.95),
            "is_valid": np.bool_(True),
            "count": np.int64(100),
            "tensor": torch.tensor(1.23)
        },
        "flags": [np.bool_(False), True, np.int32(5)]
    }

    sanitized = sanitize_numpy(nested_data)

    # Verify we can dump it to JSON
    json_str = json.dumps(sanitized)
    loaded = json.loads(json_str)

    import pytest
    assert loaded["scores"] == pytest.approx([0.9, 0.1, 0.5])
    assert loaded["metrics"]["accuracy"] == 0.95
    assert loaded["metrics"]["is_valid"] is True
    assert loaded["metrics"]["count"] == 100
    assert loaded["metrics"]["tensor"] == pytest.approx(1.23)
    assert loaded["flags"] == [False, True, 5]
