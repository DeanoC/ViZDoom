//
// Created by deano on 26/04/16.
//

#ifndef VIZDOOM_CUDACONTEXT_H
#define VIZDOOM_CUDACONTEXT_H

#include <cublas_v2.h>
#include <cudnn.h>

class CudaContext {
public:
    CudaContext(int _gpuid);

    ~CudaContext();

protected:
    const int gpuId;

    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;
};


#endif //VIZDOOM_CUDACONTEXT_H
