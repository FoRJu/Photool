"""
One thread inside numpy's OpenBLAS. Imported first by ``serve.py`` and ``main.py``:
OpenBLAS reads this once, when numpy first loads it.

Windows' OpenBLAS is not safe with several Python threads calling it at once, and
the server's requests, renders and background jobs all do. Eight threads doing
matrix multiplies brought the process down with an access violation within 40 s,
every run, and with one thread never did. In the server that looked like crashes
in OpenBLAS, ntdll and even unmapped addresses, with no Python traceback.

The cost is small: a full 24 MP multiply went from 57 ms to 125 ms, and the shape
most of the render uses, (H, W, 3), was not threaded anyway (118 ms to 124 ms). An
``OPENBLAS_NUM_THREADS`` already set is left alone.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
