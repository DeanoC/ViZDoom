//
// Created by deano on 30/04/16.
//

#include "common.h"
#include "cudainit.h"
#include "cudacontext.h"
#include <gtest/gtest.h>


TEST(CudaContextTests, valid) {
    // requires at least one gpu
    int numGpus = cudaInit();
    ASSERT_GE(numGpus, 0);
    auto ctx = cudaGetContext(0);
    ASSERT_NE(ctx, nullptr);
    ASSERT_NE(ctx->getCublasHandle(), nullptr);
    ASSERT_NE(ctx->getCudnnHandle(), nullptr);
    ASSERT_EQ(ctx->getGpuId(), 0);
    ASSERT_EQ(ctx->getWorkspaceSize(), 0);
    ctx = nullptr;
    cudaShutdown();
}

TEST(CudaContextTests, reservations) {
    cudaInit();

    auto ctx = cudaGetContext(0);
    ASSERT_NO_THROW(ctx->reserveWorkspace(256));
    ASSERT_EQ(ctx->getWorkspaceSize(), 256);
    ASSERT_NO_THROW(ctx->reserveWorkspace(256));
    ASSERT_EQ(ctx->getWorkspaceSize(), 512);
    ASSERT_NO_THROW(ctx->unreserveWorkspace(256));
    ASSERT_EQ(ctx->getWorkspaceSize(), 256);
    ASSERT_NO_THROW(ctx->unreserveWorkspace(256));
    ASSERT_EQ(ctx->getWorkspaceSize(), 0);

    ASSERT_NO_THROW(ctx->reserveWorkspace(256));
    auto ptr0 = ctx->grabWorkspace(256);
    ASSERT_NE(ptr0, nullptr);

    ASSERT_DEATH(ctx->grabWorkspace(1), "Assertion");

    ASSERT_NO_THROW(ctx->reserveWorkspace(256));
    auto ptr1 = ctx->grabWorkspace(256);
    ASSERT_NE(ptr1, nullptr);

    ASSERT_NO_FATAL_FAILURE(ctx->unreserveWorkspace(512));

    ASSERT_NO_THROW(ctx->reserveWorkspace(256));
    auto ptr2 = ctx->grabWorkspace(128);
    ASSERT_NE(ptr2, nullptr);
    auto ptr3 = ctx->grabWorkspace(128);
    ASSERT_NE(ptr3, nullptr);
    ctx->releaseWorkspace(ptr3, 128);
    auto ptr4 = ctx->grabWorkspace(128);
    ASSERT_EQ(ptr3, ptr4);


    cudaShutdown();
}