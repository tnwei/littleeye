from collections import Counter
from typing import Any

import numpy as np


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

    # Main type dispatcher
    if isinstance(obj, list):
        return _analyze_list(obj, max_depth, _current_depth)
    elif isinstance(obj, tuple):
        return _analyze_tuple(obj, max_depth, _current_depth)
    elif isinstance(obj, dict):
        return _analyze_dict(obj, max_depth, _current_depth)
    elif isinstance(obj, np.ndarray):
        return _analyze_numpy_array(obj)
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


def _analyze_numpy_array(obj: np.ndarray) -> str:
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

    # Convert to list for easier analysis
    items = list(container)

    # Check if all items are the same type
    types = [type(item).__name__ for item in items]
    type_counts = Counter(types)

    if len(type_counts) == 1:
        # Homogeneous container
        item_type = types[0]

        if item_type == "ndarray":
            return _analyze_numpy_array_collection(items)
        elif item_type in ["list", "tuple", "dict"]:
            # Nested containers - show basic info
            sizes = [len(item) for item in items]
            size_analysis = _analyze_sizes(sizes)
            return f"{item_type}s {size_analysis}"
        else:
            # Primitive types
            return f"{item_type} objects"
    else:
        # Heterogeneous container
        most_common_type, count = type_counts.most_common(1)[0]
        if count == len(items) - 1:
            return f"mostly {most_common_type} objects with 1 {_get_other_type(type_counts, most_common_type)}"
        else:
            return f"mixed types: {', '.join(f'{count} {typ}' for typ, count in type_counts.most_common())}"


def _analyze_numpy_array_collection(arrays: list[np.ndarray]) -> str:
    """Analyze a collection of numpy arrays."""
    if not arrays:
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


# Example usage and testing
if __name__ == "__main__":
    # Test cases

    # Test 1: List of numpy arrays with same shape
    test1 = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    print("Test 1:")
    print(littleeye(test1))
    print()

    # Test 2: Dictionary with sequential keys and variable numpy arrays
    test2 = {i: np.random.random(i) for i in range(1, 6)}
    print("Test 2:")
    print(littleeye(test2))
    print()

    # Test 3: Mixed container
    test3 = [1, 2, 3, "hello", [1, 2, 3]]
    print("Test 3:")
    print(littleeye(test3))
    print()

    # Test 4: Large numpy array
    test4 = np.random.random((128, 64))
    print("Test 4:")
    print(littleeye(test4))
    print()

    # Test 5: Empty containers
    test5 = {"empty_list": [], "empty_dict": {}, "numbers": [1, 2, 3]}
    print("Test 5:")
    print(littleeye(test5))
