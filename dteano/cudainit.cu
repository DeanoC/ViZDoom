//
// Created by deano on 26/04/16.
//

#include "cudainit.h"

#include <cmath>
#include <ctime>

#include <algorithm>
#include <iomanip>
#include <map>

#include <cuda_runtime.h>
#include <cassert>
#include "cudacontext.h"

namespace {
    std::vector< std::shared_ptr< class CudaContext>> cudaContexts;
    int numGpus = -1;
}

int cudaInit() {
    checkCudaErrors(cudaGetDeviceCount(&numGpus));

    for( int i = 0; i < numGpus; ++i ) {
        cudaContexts.emplace_back(std::make_shared< CudaContext >(i));
    }

    return numGpus;
}

void cudaShutdown() {
    cudaContexts.clear();
}

std::shared_ptr< class CudaContext > cudaGetContext( int gpuId ) {
    assert(gpuId < numGpus);
    return cudaContexts[ gpuId ];
}

int cudaGetContextCount() {
    return numGpus;
}