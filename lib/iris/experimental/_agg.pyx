cimport numpy as np


cdef extern from "_agg_raster.h":
    void raster(np.uint8_t *weights, const double *xi, const double *yi,
                int nx, int ny)


def _raster_weights(np.ndarray[np.uint8_t, ndim=2] weights,
                    np.ndarray[np.float64_t, ndim=2] xi,
                    np.ndarray[np.float64_t, ndim=2] yi):
    raster(<np.uint8_t *>weights.data, <const double *>xi.data,
           <const double *>yi.data, weights.shape[1], weights.shape[0])
