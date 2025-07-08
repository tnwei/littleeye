# LittleEye Design Document

## Overview
LittleEye is a Python utility function that provides human-readable structural summaries of arbitrary Python objects, with a focus on nested data structures like lists, dictionaries, and numpy arrays.

## Envision usage examples

```python
>>> littleeye(obj)
list with 120 elements
└─ numpy arrays of shape (128,) each

>>> littleeye(obj)
dict with 20 elements
└─ keys: increasing range of 23 to 43
   values: numpy arrays of variable shape of 1-d, from (3,) to (128,)
```

## Design Goals
- Generate concise, tree-like printouts of object structure
- Detect and summarize patterns in homogeneous containers
- Handle nested structures gracefully
- Provide meaningful summaries without overwhelming detail
- Focus on common data science structures (lists, dicts, numpy arrays)

## Core Architecture

### 1. Main Dispatcher
- **Function**: `littleeye(obj, max_depth=3)`
- **Purpose**: Entry point that determines object type and routes to appropriate analyzer
- **Returns**: Formatted string representation

### 2. Type-Specific Analyzers
Each analyzer handles a specific data type:

#### List/Tuple Analyzer
- Detects homogeneous vs heterogeneous content
- For homogeneous: checks if all elements have same type/shape
- For numpy arrays: analyzes shape consistency
- Handles mixed types with fallback descriptions

#### Dictionary Analyzer
- **Key Analysis**: Detects patterns in keys (sequential numbers, string patterns)
- **Value Analysis**: Separate analysis of value types and patterns
- **Consistency Checks**: Determines if values follow patterns

#### Numpy Array Analyzer
- Shape and dtype information
- Basic statistics for numeric data (optional)
- Memory usage considerations

#### Primitive Type Handler
- Simple descriptions for basic types (int, str, float, etc.)
- Set boundaries for when to show actual values vs summaries

### 3. Pattern Detection Engine
Specialized functions to identify common patterns:

#### `detect_homogeneous_type(container)`
- Checks if all elements are the same type
- Returns type and confidence level

#### `detect_shape_consistency(arrays)`
- For collections of numpy arrays
- Returns: identical, similar, or variable shapes
- Provides shape range information

#### `detect_numeric_sequence(keys)`
- Identifies if keys follow numeric patterns
- Returns range information and sequence type

#### `detect_size_patterns(objects)`
- Analyzes size distribution in collections
- Returns size ranges and patterns

### 4. Formatting Layer
Handles the tree-like output structure:

#### `format_tree_output(analysis_result)`
- Creates the └─ tree structure
- Manages indentation for nested levels
- Handles line wrapping for long descriptions

## Implementation Strategy

### Phase 1: Basic Structure (MVP)
- Main dispatcher for common types
- Simple list/dict/array analysis
- Basic formatting output
- No deep pattern detection yet

### Phase 2: Pattern Detection
- Homogeneous container detection
- Shape consistency analysis
- Numeric sequence detection
- Enhanced formatting

### Phase 3: Advanced Features
- Nested structure handling
- Custom type support
- Performance optimization
- Configuration options

## Technical Considerations

### Recursion and Depth Control
- **Max Depth**: Prevent infinite recursion in deeply nested structures
- **Circular Reference Detection**: Handle self-referencing objects
- **Performance**: Avoid analyzing extremely large structures in detail

### Memory Efficiency
- **Sampling**: For very large containers, analyze samples rather than full contents
- **Lazy Evaluation**: Only analyze what's needed for the summary
- **Threshold Management**: Define size limits for detailed analysis

### Error Handling
- **Type Safety**: Handle unknown or custom types gracefully
- **Numpy Dependencies**: Graceful fallback when numpy isn't available
- **Malformed Data**: Robust handling of incomplete or corrupted structures

## Pattern Detection Examples

### Homogeneous Arrays
```python
# Input: [np.array([1,2,3]), np.array([4,5,6]), np.array([7,8,9])]
# Output: "list with 3 elements\n└─ numpy arrays of shape (3,) each"
```

### Sequential Keys
```python
# Input: {23: arr1, 24: arr2, 25: arr3, ..., 43: arr21}
# Output: "dict with 21 elements\n└─ keys: sequential range 23 to 43"
```

### Variable Shapes
```python
# Input: [np.array([1]), np.array([1,2]), np.array([1,2,3])]
# Output: "list with 3 elements\n└─ numpy arrays of variable shape 1-d, from (1,) to (3,)"
```

## Configuration Options (Future)
- **Verbosity Levels**: Control detail level in output
- **Type Priorities**: Emphasize certain types in mixed containers
- **Custom Formatters**: Allow user-defined formatting for specific types
- **Analysis Depth**: Control how deep to analyze nested structures

## Testing Strategy
- **Unit Tests**: Each analyzer function
- **Integration Tests**: Full pipeline with complex nested structures
- **Performance Tests**: Large data structure handling
- **Edge Cases**: Empty containers, single elements, circular references

## Future Enhancements
- **Interactive Mode**: Allow drilling down into specific parts
- **Export Options**: JSON, YAML output formats
- **Visual Output**: Integration with Jupyter notebooks
- **Diff Mode**: Compare structures between two objects
