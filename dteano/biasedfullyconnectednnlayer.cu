//
// Created by deano on 27/04/16.
//

#include "biasedfullyconnectednnlayer.h"

BiasedFullyConnectedNNLayer::BiasedFullyConnectedNNLayer( int _inputSize, int _outputSize, int _batchSize ) :
        inputSize(_inputSize),
        outputSize(_outputSize),
        batchSize(_batchSize),
        weights(inputSize * outputSize),
        bias(outputSize) {

    checkCUDNN(cudnnCreateTensorDescriptor(&tensorDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(tensorDescriptor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_HALF_OR_FLOAT,
                                          batchSize,
                                          outputSize, 1, 1));

    checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDescriptor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_HALF_OR_FLOAT,
                                          1, outputSize,
                                          1, 1));
}

BiasedFullyConnectedNNLayer::~BiasedFullyConnectedNNLayer() {

    checkCUDNN(cudnnDestroyTensorDescriptor(tensorDescriptor));
    checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDescriptor));
}