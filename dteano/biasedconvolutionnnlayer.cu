//
// Created by deano on 27/04/16.
//

#include <cudnn.h>
#include "cudainit.h"
#include "cudacontext.h"
#include "biasedconvolutionnnlayer.h"

#if USE_HALF_FLOATS
#define CUDNN_DATA_HALF_OR_FLOAT CUDNN_DATA_HALF
#else
#define CUDNN_DATA_HALF_OR_FLOAT CUDNN_DATA_FLOAT
#endif

BiasedConvolutionNNLayer::BiasedConvolutionNNLayer(
        CudaContext::ptr _context,
        int _inputWidth, int _inputHeight, int _inputChannels,
        int _outputChannels, int _kernelSize, int _batchSize ) :
        ctx(_context),
        inputWidth(_inputWidth),
        inputHeight(_inputHeight),
        inputChannels(_inputChannels),
        outputChannels(_outputChannels),
        kernelSize(_kernelSize),
        batchSize(_batchSize),
        outputWidth((inputWidth - kernelSize) + 1),
        outputHeight((inputHeight - kernelSize) + 1),
        weights(inputChannels * kernelSize * kernelSize * outputChannels),
        bias(outputChannels) {

    checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(inputTensorDescriptor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_HALF_OR_FLOAT,
                                          batchSize,
                                          inputChannels, inputHeight, inputWidth));

    checkCUDNN(cudnnCreateFilterDescriptor(&filterDescriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDescriptor,
                                          CUDNN_DATA_HALF_OR_FLOAT,
                                          CUDNN_TENSOR_NCHW,
                                          outputChannels,
                                          inputChannels,
                                          kernelSize,
                                          kernelSize));

    checkCUDNN(cudnnCreateConvolutionDescriptor(&forwardConvolutionDescriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(forwardConvolutionDescriptor,
                                               0, 0, // padding
                                               1, 1, // filter stride
                                               1, 1, // scaling
                                               CUDNN_CROSS_CORRELATION));

    int n, c, h, w;
    // Find dimension of convolution output
    // get the output to account for scaling/stride/padding etc.
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(forwardConvolutionDescriptor,
                                                     inputTensorDescriptor,
                                                     filterDescriptor,
                                                     &n, &c, &h, &w));

    checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(outputTensorDescriptor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_HALF_OR_FLOAT,
                                          n, c,
                                          h, w));

    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(ctx->getCudnnHandle(),
                                                   inputTensorDescriptor,
                                                   filterDescriptor,
                                                   forwardConvolutionDescriptor,
                                                   outputTensorDescriptor,
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                   0,
                                                   &forwardConvolutionAlgorithm));

    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(ctx->getCudnnHandle(),
                                                       inputTensorDescriptor,
                                                       filterDescriptor,
                                                       forwardConvolutionDescriptor,
                                                       outputTensorDescriptor,
                                                       forwardConvolutionAlgorithm,
                                                       &forwardWorkspaceSize));

    checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDescriptor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_HALF_OR_FLOAT,
                                          1, outputChannels,
                                          1, 1));
}

BiasedConvolutionNNLayer::~BiasedConvolutionNNLayer() {

    checkCUDNN(cudnnDestroyConvolutionDescriptor(forwardConvolutionDescriptor));
    checkCUDNN(cudnnDestroyFilterDescriptor(filterDescriptor));

    checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDescriptor));
    checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDescriptor));
    checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDescriptor));

}