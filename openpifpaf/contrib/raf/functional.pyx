# cython: infer_types=True
cimport cython
from libc.math cimport exp, fabs, sqrt, fmin, fmax
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def scalar_values_3d(float[:, :, :] field, float[:] x, float[:] y, float default=-1):
    values_np = np.full((field.shape[0], x.shape[0],), default, dtype=np.float32)
    cdef float[:,:] values = values_np
    cdef float maxx = <float>field.shape[1] - 1, maxy = <float>field.shape[0] - 1

    for i in range(values.shape[1]):
        if x[i] < 0.0 or y[i] < 0.0 or x[i] > maxx or y[i] > maxy:
            continue

        values[:, i] = field[:, <Py_ssize_t>y[i], <Py_ssize_t>x[i]]

    return values_np
