//
// Created by deano on 28/04/16.
//

#include "common.h"
#include "cudainit.h"
#include <gtest/gtest.h>


TEST(CudaInitTests, whom_tests_the_tester) {
    ASSERT_EQ(true, true);
    ASSERT_NE(true, false);
}

TEST(CudaInitTests, up_and_down) {
    // requires at least one gpu
    int numGpus = cudaInit();
    ASSERT_GE(numGpus, 0);
    ASSERT_EQ(cudaGetContextCount(), numGpus);
    ASSERT_NO_THROW(cudaShutdown());
    ASSERT_EQ(cudaGetContextCount(), 0);
    int numGpus2 = cudaInit();
    ASSERT_EQ(numGpus, numGpus2);
    ASSERT_NO_THROW(cudaGetContext(0));
    EXPECT_DEATH(cudaGetContext(numGpus), "Assertion");
    ASSERT_NO_THROW(cudaShutdown());
}