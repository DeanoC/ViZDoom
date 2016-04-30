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
public:
    BiasedConvolutionNNLayer( CudaContext::ptr _context,
                              int _inputWidth, int _inputHeight, int _inputChannels,
                              int _outputChannels, int _kernelSize, int _batchSize );

    ~BiasedConvolutionNNLayer();

    void forwardPropogate( half_or_float const alpha, half_or_float const beta, half_or_float const *const x,
                           half_or_float *y ) override;

    void backPropogate( half_or_float const alpha, half_or_float const beta, half_or_float const *const x,
                        half_or_float *y ) override;

    size_t getInputCount() const override { return inputChannels * inputHeight * inputWidth; }

    size_t getWeightCount() const override { return weights.size(); }

    size_t getOutputCount() const override { return outputChannels; }

    void setWeights( half_or_float const *const in ) override;

    bool hasBias() override { return true; }

    void setBiasWeights( half_or_float const *const in ) override;

    int getKernelSize() const { return kernelSize; }

    int getInputChannels() const { return inputChannels; }

    int getInputWidth() const { return inputWidth; }

    int getInputHeight() const { return inputHeight; }

    int getOutputWidth() const { return outputWidth; }

    int getOutputHeight() const { return outputHeight; }

    int getOutputChannels() const { return outputChannels; }

    int getBatchSize() const { return batchSize; }

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
    std::vector< half_or_float > biasWeights;

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
