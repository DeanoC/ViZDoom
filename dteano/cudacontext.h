//
// Created by deano on 26/04/16.
//
#pragma once
#ifndef VIZDOOM_CUDACONTEXT_H
#define VIZDOOM_CUDACONTEXT_H

#include <cuda.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <memory>
#include <sstream>
#include <cstdint>
#include <iostream>

//////////////////////////////////////////////////////////////////////////////
// Error handling
// Adapted from the CUDNN classification code
// sample: https://developer.nvidia.com/cuDNN

#define FatalError( s ) do {                                           \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
} while(0)

#define checkCUDNN( status ) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors( status ) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#if defined(USE_HALF_FLOATS)
#define CUDNN_DATA_HALF_OR_FLOAT CUDNN_DATA_HALF
typedef half half_or_float;
#else
#define CUDNN_DATA_HALF_OR_FLOAT CUDNN_DATA_FLOAT
typedef half half_or_float;
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

    // workspace system is a simple linear buffer of device memory for temporary calculation workspace
    void reserveWorkspace( const size_t size );

    void *grabWorkspace( const size_t size );

    void releaseWorkspace( void *const ptr, const size_t size );

protected:
    const int gpuId;

    size_t maxWorkspaceRAM;
    size_t curWorkspaceOffset;
    void *workspaceBase;


    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;
};


#endif //VIZDOOM_CUDACONTEXT_H
