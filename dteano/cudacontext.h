//
// Created by deano on 26/04/16.
//

#ifndef VIZDOOM_CUDACONTEXT_H
#define VIZDOOM_CUDACONTEXT_H

#include <cublas_v2.h>
#include <cudnn.h>
#include <memory>

#if defined(USE_HALF_FLOATS)
using half_float = half;
#else
using half_float = half;
#endif

class CudaContext {
public:
    using ptr = std::shared_ptr< CudaContext >;

    CudaContext( int _gpuid );

    ~CudaContext();

    cudnnContext *getCudnnHandle() {
        return cudnnHandle;
    }

    cublasContext *getCublasHandle() {
        return cublasHandle;
    }

    int getGpuId() const {
        return gpuId;
    }

protected:
    const int gpuId;

    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;
};


#endif //VIZDOOM_CUDACONTEXT_H
