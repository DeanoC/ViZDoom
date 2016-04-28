//
// Created by deano on 27/04/16.
//
#pragma once
#ifndef VIZDOOM_BIASEDCONVOLUTIONNNLAYER_H
#define VIZDOOM_BIASEDCONVOLUTIONNNLAYER_H

#include <vector>
#include "cudacontext.h"
#include "neurallayer.h"

class BiasedConvolutionNNLayer : public NeuralLayer {

    BiasedConvolutionNNLayer( CudaContext::ptr _context,
                              int _inputWidth, int _inputHeight, int _inputChannels,
                              int _outputChannels, int _kernelSize, int _batchSize );

    ~BiasedConvolutionNNLayer();

    void forwardPropogate( const float alpha, const float beta, const float *x, float *y ) override;

    void backPropogate( const float alpha, const float beta, const float *x, float *y ) override;

protected:

    const int kernelSize;
    const int inputChannels;
    const int inputWidth;
    const int inputHeight;
    const int outputWidth;
    const int outputHeight;
    const int outputChannels;
    int batchSize;

    std::vector< half_or_float > weights;
    std::vector< half_or_float > bias;

    CudaContext::ptr ctx;
    cudnnTensorDescriptor_t inputTensorDescriptor;
    cudnnTensorDescriptor_t outputTensorDescriptor;
    cudnnTensorDescriptor_t biasTensorDescriptor;

    cudnnFilterDescriptor_t filterDescriptor;
    cudnnConvolutionDescriptor_t forwardConvolutionDescriptor;
    cudnnConvolutionFwdAlgo_t forwardConvolutionAlgorithm;
    size_t forwardWorkspaceSize;

};


#endif //VIZDOOM_BIASEDCONVOLUTIONNNLAYER_H
