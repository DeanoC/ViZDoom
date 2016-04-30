//
// Created by deano on 26/04/16.
//

#include <device_launch_parameters.h>
#include <cassert>
#include "cudainit.h"
#include "cudacontext.h"

bool gPrintCudaDeviceProperties = false;

CudaContext::CudaContext( int _gpuid ) :
        gpuId(_gpuid),
        maxWorkspaceRAM(0),
        curWorkspaceOffset(0) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, _gpuid);
    std::cout << "Device " << _gpuid << " (" << deviceProp.name << ")" <<
    " has compute capability " << deviceProp.major << "." << deviceProp.minor <<
    "\n";

    if( gPrintCudaDeviceProperties ) {
#define CUDA_PRINT_PROPERTY( PROP, COMMENT ) std::cout << #PROP << ": " << deviceProp.PROP << " ( " COMMENT" )\n"

#define CUDA_PRINT_PROPERTY_ARRAY( PROP, N, COMMENT ) { std::cout << #PROP"[" << N << "]: <";                      \
                                                      for (int i = 0; i < (N); ++i) {                            \
                                                        std::cout << deviceProp.PROP[i];                         \
                                                        std::cout << (( i < (N)-1 ) ? "," : "> ( " COMMENT" )\n");  \
                                                      }                                                          \
                                                    }

        CUDA_PRINT_PROPERTY(totalGlobalMem, "Global memory available on device in bytes");
        CUDA_PRINT_PROPERTY(sharedMemPerBlock, "Shared memory available per block in bytes");
        CUDA_PRINT_PROPERTY(regsPerBlock, "32-bit registers available per block");
        CUDA_PRINT_PROPERTY(warpSize, "Warp size in threads");
        CUDA_PRINT_PROPERTY(memPitch, "Maximum pitch in bytes allowed by memory copies");
        CUDA_PRINT_PROPERTY(maxThreadsPerBlock, "Maximum number of threads per block");
        CUDA_PRINT_PROPERTY_ARRAY(maxThreadsDim, 3, "Maximum size of each dimension of a block");
        CUDA_PRINT_PROPERTY_ARRAY(maxGridSize, 3, "Maximum size of each dimension of a grid");
        CUDA_PRINT_PROPERTY(clockRate, "Clock frequency in kilohertz");
        CUDA_PRINT_PROPERTY(totalConstMem, "Constant memory available on device in bytes");
        CUDA_PRINT_PROPERTY(major, "Major compute capability");
        CUDA_PRINT_PROPERTY(minor, "Minor compute capability");
        CUDA_PRINT_PROPERTY(textureAlignment, "Alignment requirement for textures");
        CUDA_PRINT_PROPERTY(texturePitchAlignment,
                            "Pitch alignment requirement for texture references bound to pitched memory");
        CUDA_PRINT_PROPERTY(deviceOverlap,
                            "Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount.");
        CUDA_PRINT_PROPERTY(multiProcessorCount, "Number of multiprocessors on device");
        CUDA_PRINT_PROPERTY(kernelExecTimeoutEnabled, "Specified whether there is a run time limit on kernels");
        CUDA_PRINT_PROPERTY(integrated, "Device is integrated as opposed to discrete");
        CUDA_PRINT_PROPERTY(canMapHostMemory, "Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer");
        CUDA_PRINT_PROPERTY(computeMode, "Compute mode (See ::cudaComputeMode)");
        CUDA_PRINT_PROPERTY(maxTexture1D, "Maximum 1D texture size");
        CUDA_PRINT_PROPERTY(maxTexture1DMipmap, "Maximum 1D mipmapped texture size");
        CUDA_PRINT_PROPERTY(maxTexture1DLinear, "Maximum size for 1D textures bound to linear memory");
        CUDA_PRINT_PROPERTY_ARRAY(maxTexture2D, 2, "Maximum 2D texture dimensions");
        CUDA_PRINT_PROPERTY_ARRAY(maxTexture2DMipmap, 2, "Maximum 2D mipmapped texture dimensions");
        CUDA_PRINT_PROPERTY_ARRAY(maxTexture2DLinear, 3,
                                  "Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory");
        CUDA_PRINT_PROPERTY_ARRAY(maxTexture2DGather, 2,
                                  "Maximum 2D texture dimensions if texture gather operations have to be performed");
        CUDA_PRINT_PROPERTY_ARRAY(maxTexture3D, 3, "Maximum 3D texture dimensions");
        CUDA_PRINT_PROPERTY_ARRAY(maxTexture3DAlt, 3, "Maximum alternate 3D texture dimensions");
        CUDA_PRINT_PROPERTY(maxTextureCubemap, "Maximum Cubemap texture dimensions");
        CUDA_PRINT_PROPERTY_ARRAY(maxTexture1DLayered, 2, "Maximum 1D layered texture dimensions");
        CUDA_PRINT_PROPERTY_ARRAY(maxTexture2DLayered, 3, "Maximum 2D layered texture dimensions");
        CUDA_PRINT_PROPERTY_ARRAY(maxTextureCubemapLayered, 2, "Maximum Cubemap layered texture dimensions");
        CUDA_PRINT_PROPERTY(maxSurface1D, "Maximum 1D surface size");
        CUDA_PRINT_PROPERTY_ARRAY(maxSurface2D, 2, "Maximum 2D surface dimensions");
        CUDA_PRINT_PROPERTY_ARRAY(maxSurface3D, 3, "Maximum 3D surface dimensions");
        CUDA_PRINT_PROPERTY_ARRAY(maxSurface1DLayered, 2, "Maximum 1D layered surface dimensions");
        CUDA_PRINT_PROPERTY_ARRAY(maxSurface2DLayered, 3, "Maximum 2D layered surface dimensions");
        CUDA_PRINT_PROPERTY(maxSurfaceCubemap, "Maximum Cubemap surface dimensions");
        CUDA_PRINT_PROPERTY_ARRAY(maxSurfaceCubemapLayered, 2, "Maximum Cubemap layered surface dimensions");
        CUDA_PRINT_PROPERTY(surfaceAlignment, "Alignment requirements for surfaces");
        CUDA_PRINT_PROPERTY(concurrentKernels, "Device can possibly execute multiple kernels concurrently");
        CUDA_PRINT_PROPERTY(ECCEnabled, "Device has ECC support enabled");
        CUDA_PRINT_PROPERTY(pciBusID, "PCI bus ID of the device");
        CUDA_PRINT_PROPERTY(pciDeviceID, "PCI device ID of the device");
        CUDA_PRINT_PROPERTY(pciDomainID, "PCI domain ID of the device");
        CUDA_PRINT_PROPERTY(tccDriver, "1 if device is a Tesla device using TCC driver 0 otherwise");
        CUDA_PRINT_PROPERTY(asyncEngineCount, "Number of asynchronous engines");
        CUDA_PRINT_PROPERTY(unifiedAddressing, "Device shares a unified address space with the host");
        CUDA_PRINT_PROPERTY(memoryClockRate, "Peak memory clock frequency in kilohertz");
        CUDA_PRINT_PROPERTY(memoryBusWidth, "Global memory bus width in bits");
        CUDA_PRINT_PROPERTY(l2CacheSize, "Size of L2 cache in bytes");
        CUDA_PRINT_PROPERTY(maxThreadsPerMultiProcessor, "Maximum resident threads per multiprocessor");
        CUDA_PRINT_PROPERTY(streamPrioritiesSupported, "Device supports stream priorities");
        CUDA_PRINT_PROPERTY(globalL1CacheSupported, "Device supports caching globals in L1");
        CUDA_PRINT_PROPERTY(localL1CacheSupported, "Device supports caching locals in L1");
        CUDA_PRINT_PROPERTY(sharedMemPerMultiprocessor, "Shared memory available per multiprocessor in bytes");
        CUDA_PRINT_PROPERTY(regsPerMultiprocessor, "32-bit registers available per multiprocessor");
        CUDA_PRINT_PROPERTY(managedMemory, "Device supports allocating managed memory on this system");
        CUDA_PRINT_PROPERTY(isMultiGpuBoard, "Device is on a multi-GPU board");
        CUDA_PRINT_PROPERTY(multiGpuBoardGroupID,
                            "Unique identifier for a group of devices on the same multi-GPU board");

#undef CUDA_PRINT_PROPERTY
#undef CUDA_PRINT_PROPERTY_ARRAY
    }

    // Create CUBLAS and CUDNN handles
    checkCudaErrors(cudaSetDevice(gpuId));
    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCUDNN(cudnnCreate(&cudnnHandle));
}

CudaContext::~CudaContext() {
    checkCudaErrors(cudaSetDevice(gpuId));
    checkCudaErrors(cublasDestroy(cublasHandle));
    checkCUDNN(cudnnDestroy(cudnnHandle));
}

void CudaContext::reserveWorkspace( const size_t size ) {
    curWorkspaceOffset = 0;
    maxWorkspaceRAM += size;
    checkCudaErrors(cudaFree(workspaceBase));
    checkCudaErrors(cudaMalloc((void **) &workspaceBase, size));
}

void CudaContext::unreserveWorkspace( const size_t size ) {
    curWorkspaceOffset = 0;
    maxWorkspaceRAM -= size;

    checkCudaErrors(cudaFree(workspaceBase));
    checkCudaErrors(cudaMalloc((void **) &workspaceBase, size));

}

void *CudaContext::grabWorkspace( const size_t size ) {
    assert(size + curWorkspaceOffset <= maxWorkspaceRAM);
    void *ret = ((uint8_t *) workspaceBase) + curWorkspaceOffset;
    curWorkspaceOffset = curWorkspaceOffset + size;
    return ret;
}

void CudaContext::releaseWorkspace( void *const ptr, const size_t size ) {
    assert(ptr == ((uint8_t *) workspaceBase) + curWorkspaceOffset - size);
    curWorkspaceOffset = curWorkspaceOffset - size;
}




///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels

/**
 * Fills a floating-point array with ones.
 *
 * @param vec The array to fill.
 * @param size The number of elements in the array.
 */
__global__ void FillOnes( float *vec, int size ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx >= size ) {
        return;
    }

    vec[ idx ] = 1.0f;
}

/**
 * Computes the backpropagation results of the Softmax loss for each result in a batch.
 * Uses the softmax values obtained from forward propagation to compute the difference.
 *
 * @param label The training batch label values.
 * @param num_labels The number of possible labels.
 * @param batch_size The size of the trained batch.
 * @param diff The resulting gradient.
 */
__global__ void SoftmaxLossBackprop( const float *label, int num_labels, int batch_size, float *diff ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx >= batch_size ) {
        return;
    }

    const int label_value = static_cast<int>(label[ idx ]);

    // For each item in the batch, decrease the result of the label's value by 1
    diff[ idx * num_labels + label_value ] -= 1.0f;
}