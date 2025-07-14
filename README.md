# littleeye

Peek into the structure of arbitrary nested Python objects

```python
>>> from littleeye import littleeye

>>> a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
>>> print(littleeye(a))
list with 3 elements
└─ lists of size 4 each

>>> import numpy as np
>>> b = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
>>> print(littleeye(b))
list with 3 elements
└─ numpy arrays of shape (3,) each dtype int64

>>> c = [{"id": 1, "item": "basket", "references": [4, 5, 5]}, {"id": 321, "item": "strawberries", "references": [50]},
{"id": 9, "item": "bottle"}]
>>> print(littleeye(c))
list with 3 elements
└─ dicts of variable size, from 2 to 3
```

Installation: `uv add git+https://github.com/tnwei/littleeye` or `pip install git+https://github.com/tnwei/littleeye`

Tests: `python -m pytest test/`
