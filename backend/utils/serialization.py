import json
from datetime import date, datetime

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

def sanitize_numpy(val):
    """
    Recursively sanitize any value to ensure it contains only standard Python types
    compatible with JSON serialization.
    """
    # 1. Check NumPy generic scalars and arrays first to prevent NumPy subclasses passing as primitives
    if np is not None:
        if isinstance(val, (np.bool_, bool)):  # Include bool to convert np.bool_ or any derived bool subclass
            return bool(val)
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            return float(val)
        if isinstance(val, np.ndarray):
            return [sanitize_numpy(x) for x in val.tolist()]
        if isinstance(val, np.generic):
            try:
                return sanitize_numpy(val.item())
            except Exception:
                pass

    # 2. Check PyTorch Tensors
    if torch is not None:
        if isinstance(val, torch.Tensor):
            try:
                if val.dim() == 0:
                    return sanitize_numpy(val.item())
                return [sanitize_numpy(x) for x in val.tolist()]
            except Exception:
                pass

    # 3. Standard Python collections
    if isinstance(val, dict):
        return {str(k): sanitize_numpy(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [sanitize_numpy(x) for x in val]

    # 4. Dates and datetimes
    if isinstance(val, (datetime, date)):
        return val.isoformat()

    # 5. Check for dict-like objects or Pydantic models
    if hasattr(val, "dict") and callable(val.dict):
        try:
            return sanitize_numpy(val.dict())
        except Exception:
            pass
    if hasattr(val, "to_dict") and callable(val.to_dict):
        try:
            return sanitize_numpy(val.to_dict())
        except Exception:
            pass

    # 6. Standard Python primitive conversion
    if isinstance(val, (bool, int, float, str)):
        if isinstance(val, bool):
            return bool(val)
        if isinstance(val, int):
            return int(val)
        if isinstance(val, float):
            return float(val)
        return str(val)

    # 7. Ultimate fallback: string representation to prevent crashes
    try:
        json.dumps(val)
        return val
    except (TypeError, OverflowError):
        return str(val)
