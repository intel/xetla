/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include <algorithm>
#include "tests/utils/utils.hpp"
#include "xetla.hpp"

// flush cache 0: NO flush
// flush cache 1: memset
// flush cache 2: pingpong moving ptr offset
#define FLUSH_CACHE 2

using namespace cl::sycl;
using namespace gpu::xetla;
using namespace gpu;
using namespace gpu::xetla::kernel;
using namespace gpu::xetla::subgroup;

template <typename data_type_a, typename data_type_b, typename data_type_c,
        typename data_type_acc = float>
int gemm_relu_result_validate(data_type_a *A_device, data_type_b *B_device,
        data_type_c *C_device, uint32_t m, uint32_t k, uint32_t n,
        sycl::queue &queue, mem_layout mem_layout_a_ = mem_layout::row_major,
        mem_layout mem_layout_b_ = mem_layout::row_major) {
    auto A = alloc_host_and_copy<data_type_a>(A_device, m * k, queue);
    auto B = alloc_host_and_copy<data_type_b>(B_device, k * n, queue);
    auto C = alloc_host_and_copy<data_type_c>(C_device, m * n, queue);

    buff_cmp::buff_vals<data_type_c> data(C, m, n, n);
    std::vector<data_type_acc> gold_C(m * n, 0);
    get_gemm_gold<data_type_a, data_type_b, data_type_acc>(
            m, n, k, mem_layout_a_, mem_layout_b_, A, B, gold_C.data());
    // ReLU
    std::transform(gold_C.cbegin(), gold_C.cend(), gold_C.begin(),
            [](data_type_acc c) { return c > 0.0f ? c : 0.0f; });

    buff_cmp::buff_vals<data_type_c, data_type_acc> other(
            gold_C.data(), m, n, n);

    bool result = buff_cmp::xetla_buff_cmp(data, other, "gemm_relu validation");

    free(A);
    free(B);
    free(C);

    std::cout << (!result ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}

template <uint32_t wg_tile_m, uint32_t wg_tile_n, uint32_t sg_tile_m,
        uint32_t sg_tile_n, uint32_t sg_tile_k>
void gemm_relu(uint32_t matrix_m, uint32_t matrix_k, uint32_t matrix_n,
        size_t batch_num = 1) {
    // Tips, the example demonstrates programming kernel with XeTLA, it works as expected with current configurations.
    // Please make sure you fully understand these configurations before you do any modifications, incomplete changes may lead to unexpected behaviors.
    // Please contact us for support.

    //GEMM input size
    uint32_t size_a = matrix_m * matrix_k;
    uint32_t size_b = matrix_k * matrix_n;
    uint32_t size_c = matrix_m * matrix_n;

    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = float;
    using data_type_d = float;
    using data_type_acc = float;

    static constexpr auto iter = 10;
    static constexpr auto l3_cache_size = 256 * 1024 * 1024;
    static constexpr auto pingpong_flush_iter = 3;

    //Turn on the profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling()};

    //Define SYCL queue, context and device
    auto queue = sycl::queue(properties);
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

#if FLUSH_CACHE == 1
    auto dev_cache = alloc_device_and_init<int8_t>(
            l3_cache_size,
            [](int8_t *data, size_t idx) {
                data[idx] = static_cast<int8_t>(random_float());
            },
            queue, device, context);
#endif

#if FLUSH_CACHE == 2
    size_t pingpong_size_a
            = max(batch_num * size_a, l3_cache_size / sizeof(data_type_a));
    size_t pingpong_size_b
            = max(batch_num * size_b, l3_cache_size / sizeof(data_type_b));
    size_t pingpong_size_c
            = max(batch_num * size_c, l3_cache_size / sizeof(data_type_c));

    auto A = alloc_device_and_init<data_type_a>(
            pingpong_size_a * pingpong_flush_iter,
            [](data_type_a *data, size_t idx) {
                data[idx] = static_cast<data_type_a>(random_float());
            },
            queue, device, context);
    auto B = alloc_device_and_init<data_type_b>(
            pingpong_size_b * pingpong_flush_iter,
            [](data_type_b *data, size_t idx) {
                data[idx] = static_cast<data_type_b>(random_float());
            },
            queue, device, context);
    auto C = alloc_device_and_init<data_type_c>(
            pingpong_size_c * pingpong_flush_iter,
            [](data_type_c *data, size_t idx) {
                data[idx] = static_cast<data_type_c>(0.0f);
            },
            queue, device, context);
#else
    auto A = alloc_device_and_init<data_type_a>(
            batch_num * size_a,
            [](data_type_a *data, size_t idx) {
                data[idx] = static_cast<data_type_a>(random_float());
            },
            queue, device, context);
    auto B = alloc_device_and_init<data_type_b>(
            batch_num * size_b,
            [](data_type_b *data, size_t idx) {
                data[idx] = static_cast<data_type_b>(random_float());
            },
            queue, device, context);
    auto C = alloc_device_and_init<data_type_c>(
            batch_num * size_c,
            [](data_type_c *data, size_t idx) {
                data[idx] = static_cast<data_type_c>(0.0f);
            },
            queue, device, context);
#endif

    //Define the shape of workgroup
    //It's tunable parameters based on different input shape and hardware for better performance
    static constexpr uint32_t periodic_sync_interval = 8;
    static constexpr uint32_t prefetch_distance = 3;

    // [ReLu] Chain multiple elementwise op in chained_tile_op_t<>: relu_op_t
    using tile_op_t = xetla::subgroup::chained_tile_op_t<
            xetla::subgroup::relu_op_t // apply elementwise ReLU
            >;
    // [ReLu] epilogue_t is an elementwise operation that will be applied to the
    // accumulator C_acc in the final stage, in which
    //   C_acc = A x B
    // is already calculated.
    // Mathematically epilogue_t is a map that applies to each element:
    //   epilogue_t: [m, n] -> [m, n], C_acc |-> tile_op_t(C_acc)
    using epilogue_policy
            = xetla::group::epilogue_policy_tile_op<tile_op_t, gpu_arch::Xe>;
    using tile_shape = gpu::xetla::group::tile_shape_t<wg_tile_n, wg_tile_m,
            sg_tile_n, sg_tile_m>;

    // Mirco-kernel configuration
    using gemm_t = typename xetla::group::gemm_selector_t<data_type_a,
            data_type_b, mem_layout::row_major, mem_layout::row_major,
            mem_space::global, mem_space::global, 8, 8, data_type_acc,
            tile_shape, sg_tile_k, mma_engine::xmx, gpu_arch::Xe,
            prefetch_distance, periodic_sync_interval>::gemm;

    using epilogue_t = xetla::group::epilogue_t<epilogue_policy, tile_shape,
            mem_desc_t<data_type_c, mem_layout::row_major, mem_space::global>>;

    using group_swizzle = xetla::kernel::group_swizzle_default<gpu_arch::Xe>;

    using dispatch_policy = dispatch_policy_kslicing<group_swizzle, 1, 1>;

    using gemm_op_t = gemm_universal_t<dispatch_policy, gemm_t, epilogue_t>;

    typename gemm_op_t::arguments_t arg(matrix_m, matrix_k, matrix_n, nullptr,
            matrix_k, nullptr, matrix_n, nullptr, matrix_n, nullptr, nullptr);

    cl::sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(arg);

    constexpr uint32_t warmup = 10;
    long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k
            + matrix_m * matrix_n;
    profiling_helper prof("gemm_relu_run", ops, "gflops");
    long ops_hbm = (sizeof(data_type_a) * matrix_m * matrix_k
                           + sizeof(data_type_b) * matrix_k * matrix_n
                           + sizeof(data_type_c) * matrix_m * matrix_n)
            * batch_num;
    profiling_helper prof_hbm("gemm_relu_run", ops_hbm, "GB/s");
    for (uint32_t i = 0; i < iter + warmup; i++) {
#if FLUSH_CACHE == 1
        queue.memset((void *)(dev_cache), 0, l3_cache_size).wait();
#endif

        if (i >= warmup) {
            prof.cpu_start();
            prof_hbm.cpu_start();
        }
        auto gpu_event = queue.submit([&](handler &cgh) {
            // GPU kernel
            cgh.parallel_for(nd_range, [=](nd_item<3> item) KERNEL_MAIN {
                // allocate slm and nbarrier resource
                slm_barrier_init<gemm_op_t>();
                gemm_op_t gemm_op;
                typename gemm_op_t::arguments_t arg(matrix_m, matrix_k,
                        matrix_n, nullptr, matrix_k, nullptr, matrix_n, nullptr,
                        matrix_n, nullptr, nullptr);
#if FLUSH_CACHE == 2
                arg.matA_base
                        = (A + (i % pingpong_flush_iter) * pingpong_size_a);
                arg.matB_base
                        = (B + (i % pingpong_flush_iter) * pingpong_size_b);
                arg.matC_base
                        = (C + (i % pingpong_flush_iter) * pingpong_size_c);
#else
                arg.matA_base
                        = A ;
                arg.matB_base
                        = B ;
                arg.matC_base
                        = C ;
#endif
                gemm_op(item, arg);
            });
        });
        gpu_event.wait();

        if (i >= warmup) {
            prof.cpu_end();
            prof.add_gpu_event(gpu_event);
            prof_hbm.cpu_end();
            prof_hbm.add_gpu_event(gpu_event);
        }
    }

    ASSERT_EQ(0,
            gemm_relu_result_validate(A, B, C, matrix_m, matrix_k, matrix_n,
                    queue, mem_layout::row_major, mem_layout::row_major));

    //performance
    prof.print_profiling_result(profiling_selector::GPU);
    prof_hbm.print_profiling_result(profiling_selector::GPU);

    free(A, context);
    free(B, context);
    free(C, context);
#if FLUSH_CACHE == 1
    free(dev_cache, context);
#endif
}

int main() {
    // The purpose of this example is to illustrate the epilogue_t API in XeTLA.

    // It allows user to implement multiple Ops inside a kernel call to avoid
    // overheads in invokation, memory transfer, etc.
    // Take the following python code as an example:

    // Original:
    // > import torch as to
    // > x = to.matmul(A, B)
    // > y = to.nn.functional.relu(x)

    // It takes two kernel invokations and the ReLU Op is a elementwise operation
    // that could be fused into MatMul Op, which is basically calling GEMM kernel.

    // Fusion:
    // > import torch as to
    // > y = MatMulReLU(A, B)
    // The MatMulReLU Op corresponds to the second example presented below.

    // It allows the user to apply custom operations in GEMM computation.
    // Here provides some possible configurations using epilogue_t:
    // - GEMM
    //   C  = A x B
    // - tile_op_t=relu_op_t
    //   C = ReLU(A x B)
    // - tile_op_t=[relu_op_t]
    //   C
    //     = ReLU(A x B)
    //  where:
    //    shape(A) = [m, k]
    //    shape(B) = [k, n]
    //    shape(C) = [m, n]
    //    shape(D) = [1, n]
    // This example will implement the last variant that chains multiple
    // operations, which demonstrates its maximal flexibility.
    // checkout op_functor.hpp for more elementwise ops

    // Note:
    //   - comments related to this example will be prefixed with "[ReLu]"
    gemm_relu<256, 256, 32, 64, 32>(4096, 4096, 4096);
    gemm_relu<256, 256, 32, 64, 32>(8192, 8192, 8192);
    gemm_relu<8, 512, 8, 16, 16>(1, 5120, 13824);
    gemm_relu<64, 512, 32, 32, 16>(1024, 28672, 8192);
    gemm_relu<256, 256, 32, 64, 32>(3072, 4096, 3072);
    gemm_relu<8, 512, 8, 16, 16>(4, 4096, 12288);
    return (0);
}
