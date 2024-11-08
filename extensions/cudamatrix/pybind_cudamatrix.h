
#ifndef KALPY_PYBIND_CUDAMATRIX_H_
#define KALPY_PYBIND_CUDAMATRIX_H_

#include "base/kaldi-error.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "pybind/kaldi_pybind.h"

void init_cudamatrix(py::module &);
#endif  // KALPY_PYBIND_CUDAMATRIX_H_
