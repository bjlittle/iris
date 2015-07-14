#ifndef _IRIS_AGG_WRAPPER
#define _IRIS_AGG_WRAPPER

#include <stdint.h>

void raster(uint8_t *weights, const double *xi, const double *yi,
            int nx, int ny);

#endif
