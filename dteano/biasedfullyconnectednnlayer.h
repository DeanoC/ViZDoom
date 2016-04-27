//
// Created by deano on 27/04/16.
//
#pragma once
#ifndef VIZDOOM_BIASEDFULLYCONNECTEDNNLAYER_H
#define VIZDOOM_BIASEDFULLYCONNECTEDNNLAYER_H

#include <vector>
#include "cudacontext.h"

class BiasedFullyConnectedNNLayer {
public:
    BiasedFullyConnectedNNLayer( int _inputSize, int _outputSize, int _batchSize );

    ~BiasedFullyConnectedNNLayer();

protected:
    const int inputSize;
    const int outputSize;
    int batchSize;

    std::vector< half_or_float > weights;
    std::vector< half_or_float > bias;

    cudnnTensorDescriptor_t tensorDescriptor;
    cudnnTensorDescriptor_t biasTensorDescriptor;

};


#endif //VIZDOOM_BIASEDFULLYCONNECTEDNNLAYER_H
