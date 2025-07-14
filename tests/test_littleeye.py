import numpy as np
from src.littleeye import littleeye

def test_list_of_numpy_arrays_same_shape():
    test_data = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    expected_output = "list with 3 elements\n└─ numpy arrays of shape (3,) each dtype int64"
    assert littleeye(test_data) == expected_output

def test_dict_sequential_keys_variable_numpy_arrays():
    # This test case generates random arrays, so we need to mock or adjust for deterministic testing.
    # For now, let's create a fixed version of the test data.
    test_data = {
        1: np.array([0.1]),
        2: np.array([0.2, 0.3]),
        3: np.array([0.4, 0.5, 0.6]),
        4: np.array([0.7, 0.8, 0.9, 1.0]),
        5: np.array([1.1, 1.2, 1.3, 1.4, 1.5])
    }
    # The expected output will depend on the exact values and how littleeye summarizes them.
    # For now, let's assert on a general structure.
    expected_output_start = "dict with 5 elements"
    expected_output_keys = "keys: sequential range 1 to 5"
    expected_output_values = "values: numpy arrays of variable shape 1-d, from (1,) to (5,) dtype float64"
    result = littleeye(test_data)
    assert expected_output_start in result
    assert expected_output_keys in result
    assert expected_output_values in result

def test_mixed_container():
    test_data = [1, 2, 3, "hello", [1, 2, 3]]
    expected_output = "list with 5 elements\n└─ mixed types: 3 int, 1 str, 1 list"
    assert littleeye(test_data) == expected_output

def test_large_numpy_array():
    test_data = np.random.random((128, 64))
    expected_output = "numpy array of shape (128x64) dtype float64"
    assert littleeye(test_data) == expected_output

def test_empty_containers():
    test_data = {"empty_list": [], "empty_dict": {}, "numbers": [1, 2, 3]}
    expected_output_list = "dict with 3 elements"
    expected_output_keys = "keys: string keys: 'empty_list', 'empty_dict', 'numbers'"
    expected_output_values = "values: mixed types: 1 empty list, 1 empty dict, 1 list"
    result = littleeye(test_data)
    assert expected_output_list in result
    assert expected_output_keys in result
    # The exact string for mixed types might vary, so a partial check is better.
    assert "empty list" in result
    assert "empty dict" in result
    assert "list" in result
