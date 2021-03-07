# cython: infer_types=True
cimport cython
from libc.math cimport exp, fabs, sqrt, fmin, fmax
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void cumulative_average_2d(float[:, :] cuma, float[:, :] cumw, float[:] x, float[:] y, float[:] width, float[:] height, float[:] v, float[:] w) nogil:
    cdef long minx, miny, maxx, maxy
    cdef float cv, cw, cx, cy, cwidth, cheight
    cdef Py_ssize_t i, xx, yy

    for i in range(x.shape[0]):
        cw = w[i]
        if cw <= 0.0:
            continue

        cv = v[i]
        cx = x[i]
        cy = y[i]
        cwidth = width[i]
        cheight = height[i]

        minx = (<long>clip(cx - cwidth, 0, cuma.shape[1] - 1))
        maxx = (<long>clip(cx + cwidth, minx + 1, cuma.shape[1]))
        miny = (<long>clip(cy - cheight, 0, cuma.shape[0] - 1))
        maxy = (<long>clip(cy + cheight, miny + 1, cuma.shape[0]))
        for xx in range(minx, maxx):
            for yy in range(miny, maxy):
                cuma[yy, xx] = (cw * cv + cumw[yy, xx] * cuma[yy, xx]) / (cumw[yy, xx] + cw)
                cumw[yy, xx] += cw

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void scalar_square_add_2dgauss_with_max(float[:, :] field, float[:] x, float[:] y, float[:] sigma_w, float[:] sigma_h, float[:] v, float truncate=2.0, float max_value=1.0) nogil:
    cdef Py_ssize_t i, xx, yy
    cdef float vv, deltax2, deltay2
    cdef float cv, cx, cy, csigma_w, csigma_w2, csigma_h, csigma_h2
    cdef long minx, miny, maxx, maxy
    cdef float truncate2 = truncate * truncate

    for i in range(x.shape[0]):
        csigma_w = sigma_w[i]
        csigma_w2 = csigma_w * csigma_w
        csigma_h = sigma_h[i]
        csigma_h2 = csigma_h * csigma_h
        cx = x[i]
        cy = y[i]
        cv = v[i]

        minx = (<long>clip(cx - truncate * csigma_w, 0, field.shape[1] - 1))
        maxx = (<long>clip(cx + truncate * csigma_w, minx + 1, field.shape[1]))
        miny = (<long>clip(cy - truncate * csigma_h, 0, field.shape[0] - 1))
        maxy = (<long>clip(cy + truncate * csigma_h, miny + 1, field.shape[0]))
        for xx in range(minx, maxx):
            deltax2 = (xx - cx)**2
            for yy in range(miny, maxy):
                deltay2 = (yy - cy)**2

                if deltax2/csigma_w2 + deltay2/csigma_h2 > truncate2:
                    continue

                if deltax2 < 0.25 and deltay2 < 0.25:
                    # this is the closest pixel
                    vv = cv
                else:
                    vv = cv * approx_exp(-0.5 * (deltax2/csigma_w2 + deltay2/csigma_h2))
                field[yy, xx] += vv
                field[yy, xx] = min(max_value, field[yy, xx])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void scalar_square_add_2dgauss(float[:, :] field, float[:] x, float[:] y, float[:] sigma_w, float[:] sigma_h, float[:] v, float truncate=2.0) nogil:
    cdef Py_ssize_t i, xx, yy
    cdef float vv, deltax2, deltay2
    cdef float cv, cx, cy, csigma_w, csigma_w2, csigma_h, csigma_h2
    cdef long minx, miny, maxx, maxy

    for i in range(x.shape[0]):
        csigma_w = sigma_w[i]
        csigma_w2 = csigma_w * csigma_w
        csigma_h = sigma_h[i]
        csigma_h2 = csigma_h * csigma_h
        cx = x[i]
        cy = y[i]
        cv = v[i]

        minx = (<long>clip(cx - truncate * csigma_w, 0, field.shape[1] - 1))
        maxx = (<long>clip(cx + truncate * csigma_w, minx + 1, field.shape[1]))
        miny = (<long>clip(cy - truncate * csigma_h, 0, field.shape[0] - 1))
        maxy = (<long>clip(cy + truncate * csigma_h, miny + 1, field.shape[0]))
        for xx in range(minx, maxx):
            deltax2 = (xx - cx)**2
            for yy in range(miny, maxy):
                deltay2 = (yy - cy)**2
                if deltax2 < 0.25 and deltay2 < 0.25:
                    # this is the closest pixel
                    vv = cv
                else:
                    vv = cv * approx_exp(-0.5 * (deltax2/csigma_w2 + deltay2/csigma_h2))
                field[yy, xx] += vv


cdef inline float clip(float v, float minv, float maxv) nogil:
    return fmax(minv, fmin(maxv, v))


cdef inline float approx_exp(float x) nogil:
    if x > 2.0 or x < -2.0:
        return 0.0
    x = 1.0 + x / 8.0
    x *= x
    x *= x
    x *= x
    return x
