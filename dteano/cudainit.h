//
// Created by deano on 26/04/16.
//

#ifndef VIZDOOM_CUDAINIT_H
#define VIZDOOM_CUDAINIT_H

#include <memory>
#include <sstream>
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

int cudaInit();

void cudaShutdown();

int cudaGetContextCount();

std::shared_ptr< class CudaContext > cudaGetContext( int gpuId );

#endif //VIZDOOM_CUDAINIT_H
