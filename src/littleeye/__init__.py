from collections import Counter
from typing import Any

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


def littleeye(obj: Any, max_depth: int = 3, _current_depth: int = 0) -> str:
    """
    Generate a human-readable structural summary of any Python object.

    Args:
        obj: The object to analyze
        max_depth: Maximum recursion depth
        _current_depth: Internal parameter for recursion tracking

    Returns:
        Formatted string representation of the object structure
    """
    if _current_depth >= max_depth:
        return f"{type(obj).__name__} (max depth reached)"

    if NUMPY_AVAILABLE:
        if np is not None and isinstance(obj, np.ndarray):
            return _analyze_numpy_array(obj)
        
    if isinstance(obj, list):
        return _analyze_list(obj, max_depth, _current_depth)
    elif isinstance(obj, tuple):
        return _analyze_tuple(obj, max_depth, _current_depth)
    elif isinstance(obj, dict):
        return _analyze_dict(obj, max_depth, _current_depth)
    elif isinstance(obj, (int, float, str, bool)):
        return _analyze_primitive(obj)
    else:
        return f"{type(obj).__name__} object"


def _analyze_list(obj: list, max_depth: int, current_depth: int) -> str:
    """Analyze list structure and contents."""
    if not obj:
        return "empty list"

    length = len(obj)
    result = f"list with {length} element{'s' if length != 1 else ''}"

    # Analyze content patterns
    content_analysis = _analyze_container_contents(obj, max_depth, current_depth)
    if content_analysis:
        result += f"\n└─ {content_analysis}"

    return result


def _analyze_tuple(obj: tuple, max_depth: int, current_depth: int) -> str:
    """Analyze tuple structure and contents."""
    if not obj:
        return "empty tuple"

    length = len(obj)
    result = f"tuple with {length} element{'s' if length != 1 else ''}"

    # Analyze content patterns
    content_analysis = _analyze_container_contents(obj, max_depth, current_depth)
    if content_analysis:
        result += f"\n└─ {content_analysis}"

    return result


def _analyze_dict(obj: dict, max_depth: int, current_depth: int) -> str:
    """Analyze dictionary structure and contents."""
    if not obj:
        return "empty dict"

    length = len(obj)
    result = f"dict with {length} element{'s' if length != 1 else ''}"

    # Analyze keys
    key_analysis = _analyze_dict_keys(obj.keys())
    if key_analysis:
        result += f"\n└─ keys: {key_analysis}"

    # Analyze values
    value_analysis = _analyze_container_contents(obj.values(), max_depth, current_depth)
    if value_analysis:
        prefix = "   " if key_analysis else "└─ "
        result += f"\n{prefix}values: {value_analysis}"

    return result


def _analyze_numpy_array(obj: Any) -> str: # Changed type hint from np.ndarray to Any
    """Analyze numpy array structure."""
    shape_str = "x".join(map(str, obj.shape))
    dtype_str = str(obj.dtype)

    if obj.ndim == 1:
        return f"numpy array of shape ({shape_str},) dtype {dtype_str}"
    else:
        return f"numpy array of shape ({shape_str}) dtype {dtype_str}"


def _analyze_primitive(obj: int | float | str | bool) -> str:
    """Analyze primitive types."""
    if isinstance(obj, str):
        if len(obj) > 50:
            return f"string of length {len(obj)}"
        else:
            return f"string: '{obj}'"
    else:
        return f"{type(obj).__name__}: {obj}"


def _analyze_container_contents(container, max_depth: int, current_depth: int) -> str:
    """Analyze the contents of a container (list, tuple, or dict values)."""
    if not container:
        return ""

    items = list(container)

    # Generate more specific descriptions for each item
    item_descriptions = [_get_simplified_type_description(item) for item in items]
    type_counts = Counter(item_descriptions)

    if len(type_counts) == 1:
        # Homogeneous container based on simplified descriptions
        item_type = item_descriptions[0]

        if item_type == "numpy array":
            return _analyze_numpy_array_collection(items)
        elif item_type.startswith("empty"):
            return f"{item_type}s" # e.g., "empty lists"
        elif item_type in ["list", "tuple", "dict"]:
            # Nested containers - show basic info
            sizes = [len(item) for item in items]
            size_analysis = _analyze_sizes(sizes)
            return f"{item_type}s {size_analysis}"
        else:
            # Primitive types
            return f"{item_type} objects"
    else:
        # Heterogeneous container based on simplified descriptions
        # Always use the verbose mixed types output
        return f"mixed types: {', '.join(f'{count} {typ}' for typ, count in type_counts.most_common())}"


def _analyze_numpy_array_collection(arrays: list[Any]) -> str: # Changed type hint to list[Any] for clarity
    """Analyze a collection of numpy arrays."""
    if not NUMPY_AVAILABLE or not arrays:
        return "no arrays"

    shapes = [arr.shape for arr in arrays]
    dtypes = [arr.dtype for arr in arrays]

    # Check shape consistency
    unique_shapes = list(set(shapes))
    unique_dtypes = list(set(dtypes))

    dtype_str = f" dtype {unique_dtypes[0]}" if len(unique_dtypes) == 1 else ""

    if len(unique_shapes) == 1:
        # All same shape
        shape = unique_shapes[0]
        if len(shape) == 1:
            return f"numpy arrays of shape ({shape[0]},) each{dtype_str}"
        else:
            shape_str = "x".join(map(str, shape))
            return f"numpy arrays of shape ({shape_str}) each{dtype_str}"
    else:
        # Variable shapes
        if all(len(shape) == 1 for shape in shapes):
            # All 1D but different lengths
            min_size = min(shape[0] for shape in shapes)
            max_size = max(shape[0] for shape in shapes)
            return f"numpy arrays of variable shape 1-d, from ({min_size},) to ({max_size},){dtype_str}"
        else:
            # Mixed dimensionality
            return f"numpy arrays of variable shapes{dtype_str}"


def _analyze_dict_keys(keys) -> str:
    """Analyze dictionary keys for patterns."""
    key_list = list(keys)

    if not key_list:
        return ""

    # Check if all keys are integers
    if all(isinstance(k, int) for k in key_list):
        sorted_keys = sorted(key_list)
        if len(sorted_keys) > 1:
            # Check if sequential
            if all(
                sorted_keys[i] == sorted_keys[i - 1] + 1
                for i in range(1, len(sorted_keys))
            ):
                return f"sequential range {sorted_keys[0]} to {sorted_keys[-1]}"
            else:
                return f"integer range {sorted_keys[0]} to {sorted_keys[-1]}"
        else:
            return f"single integer key {sorted_keys[0]}"
    elif all(isinstance(k, str) for k in key_list):
        if len(key_list) <= 5:
            return f"string keys: {', '.join(repr(k) for k in key_list)}"
        else:
            return f"string keys (showing first 3): {', '.join(repr(k) for k in key_list[:3])}, ..."
    else:
        return "mixed key types"


def _analyze_sizes(sizes: list[int]) -> str:
    """Analyze size distribution in containers."""
    if not sizes:
        return ""

    unique_sizes = list(set(sizes))

    if len(unique_sizes) == 1:
        size = unique_sizes[0]
        return f"of size {size} each"
    else:
        min_size = min(sizes)
        max_size = max(sizes)
        return f"of variable size, from {min_size} to {max_size}"


def _get_other_type(type_counts: Counter, exclude_type: str) -> str:
    """Get the other type in a mostly homogeneous container."""
    for typ, count in type_counts.items():
        if typ != exclude_type:
            return typ
    return "other"


def _get_simplified_type_description(obj: Any) -> str:
    """Returns a simplified type description for an object."""
    if isinstance(obj, list):
        return "empty list" if not obj else "list"
    elif isinstance(obj, dict):
        return "empty dict" if not obj else "dict"
    elif isinstance(obj, tuple):
        return "empty tuple" if not obj else "tuple"
    elif NUMPY_AVAILABLE and np is not None and isinstance(obj, np.ndarray):
        return "numpy array"
    else:
        return type(obj).__name__
