//
// Created by deano on 28/04/16.
//

#include "common.h"
#include "cudainit.h"
#include "cudacontext.h"
#include "neuralnetwork.h"
#include <gtest/gtest.h>

namespace {

    class DummyLayer : public NeuralLayer {
    public:
        void forwardPropogate( const float alpha, const float beta, const float *x, float *y ) override { }

        void backPropogate( const float alpha, const float beta, const float *x, float *y ) override { }

    };
}

class NeuralNetworkTest : public ::testing::Test {
protected:
    virtual void SetUp() override {
        cudaInit();
        ctx = cudaGetContext(0);
        ann1 = std::make_shared< NeuralNetwork >(1);
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

TEST_F(NeuralNetworkTest, single_layer) {
    (*ann1)[ 0 ] = std::make_shared< DummyLayer >();
    ASSERT_NE((*ann1)[ 0 ], nullptr);

}
