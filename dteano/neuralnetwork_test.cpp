//
// Created by deano on 28/04/16.
//

#include "common.h"
#include "cudainit.h"
#include "cudacontext.h"
#include "neuralnetwork.h"
#include "biasedconvolutionnnlayer.h"
#include <gtest/gtest.h>


class NeuralNetworkTest : public ::testing::Test {
protected:
    virtual void SetUp() override {
        cudaInit();
        ctx = cudaGetContext(0);
        ann1 = std::make_shared< NeuralNetwork >(ctx, 1);
    }

    virtual void TearDown() override {
        ctx.reset();
        cudaShutdown();
    }

    NeuralNetwork::ptr ann1;

    CudaContext::ptr ctx;
};

TEST_F(NeuralNetworkTest, is_empty) {
    ASSERT_NE(ann1, nullptr);

    ASSERT_NE(ann1->begin(), ann1->end());
    ASSERT_NE(ann1->cbegin(), ann1->cend());

    ASSERT_EQ((*ann1)[ 0 ], nullptr);
}

TEST_F(NeuralNetworkTest, single_biasedconvolutionnnlayer) {
    auto bcnnLayer = std::make_shared< BiasedConvolutionNNLayer >(
            ctx,
            10, 10, 1,
            1, 10, 100
    );
    (*ann1)[ 0 ] = bcnnLayer;

    ASSERT_NE((*ann1)[ 0 ], nullptr);
    ASSERT_EQ(bcnnLayer->getOutputWidth(), 1);
    ASSERT_EQ(bcnnLayer->getOutputHeight(), 1);
    std::vector< half_or_float > weights(bcnnLayer->getWeightCount(), 0);
    std::vector< half_or_float > biasWeights(bcnnLayer->getOutputCount(), 0);

    ASSERT_EQ(bcnnLayer->getWeightCount(), ann1->getWeightCount());
    std::fill(weights.begin(), weights.end(), half_or_float(1));
    std::fill(biasWeights.begin(), biasWeights.end(), half_or_float(1));

    ASSERT_NO_THROW(ann1->setWeights(weights.data()));
    ASSERT_TRUE(bcnnLayer->hasBias());
    ASSERT_NO_THROW(ann1->setBiasWeights(biasWeights.data()));


    half_or_float *x;
    half_or_float *y;

    cudaMalloc((void **) &x, ann1->getInputCount());
    cudaMalloc((void **) &y, ann1->getOutputCount());

    ASSERT_NO_THROW(ann1->presentInput(x, y));

}