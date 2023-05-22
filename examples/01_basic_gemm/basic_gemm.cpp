﻿/*******************************************************************************
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
#include "tests/utils/utils.hpp"
#include "xetla.hpp"

using namespace cl::sycl;
using namespace gpu::xetla;
using namespace gpu;

template <typename data_type_a, typename data_type_b, typename data_type_c,
        typename data_type_acc = float>
int gemm_result_validate(data_type_a *A, data_type_b *B, data_type_c *C,
        uint32_t m, uint32_t k, uint32_t n,
        mem_layout mem_layout_a_ = mem_layout::row_major,
        mem_layout mem_layout_b_ = mem_layout::row_major) {
    bool is_col_major_a = mem_layout_a_ == mem_layout::col_major;
    bool is_col_major_b = mem_layout_b_ == mem_layout::col_major;
    buff_cmp::buff_vals<data_type_c> data(C, m, n, n);
    std::vector<data_type_acc> gold_C(m * n, 0);
    get_gemm_gold<data_type_a, data_type_b, data_type_acc>(
            m, n, k, mem_layout_a_, mem_layout_b_, A, B, gold_C.data());
    buff_cmp::buff_vals<data_type_c, data_type_acc> other(
            gold_C.data(), m, n, n);

    bool result
            = buff_cmp::xetla_buff_cmp(data, other, "basic_gemm validation");

    std::cout << (!result ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}

void basic_gemm_run(uint32_t iter) {
    // Tips, the example demonstrates programming kernel with XeTLA, it works as expected with current configurations.
    // Please make sure you fully understand these configurations before you do any modifications, incomplete changes may lead to unexpected behaviors.
    // Please contact us for support.

    //GEMM input size
    uint32_t matrix_m = 4096;
    uint32_t matrix_n = 4096;
    uint32_t matrix_k = 4096;

    uint32_t size_a = matrix_m * matrix_k;
    uint32_t size_b = matrix_k * matrix_n;
    uint32_t size_c = matrix_m * matrix_n;

    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = bf16;
    using data_type_acc = float;

    //Turn on the profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling()};

    //Define SYCL queue, context and device
    auto Queue = queue(properties);
    auto Context = Queue.get_info<info::queue::context>();
    auto Device = Queue.get_info<info::queue::device>();

    std::cout << "Running on " << Device.get_info<info::device::name>() << "\n";

    //Define and initialize the data required for the calculation
    //Use shared data which will be migrated automatically between  both CPU and GPU
    data_type_a *A = static_cast<data_type_a *>(
            malloc_shared(size_a * sizeof(data_type_a), Device, Context));
    data_type_b *B = static_cast<data_type_b *>(
            malloc_shared(size_b * sizeof(data_type_b), Device, Context));
    data_type_c *C = static_cast<data_type_c *>(
            malloc_shared(size_c * sizeof(data_type_c), Device, Context));

    //Init data in GEMM A, B and C
    for (uint32_t i = 0; i < size_a; ++i) {
        A[i] = static_cast<data_type_a>(random_float());
    }
    for (uint32_t i = 0; i < size_b; ++i) {
        B[i] = static_cast<data_type_b>(random_float());
    }
    for (uint32_t i = 0; i < size_c; ++i) {
        C[i] = static_cast<data_type_c>(0.0f);
    }

    //Define the shape of workgroup and subgroup
    //It's tunable parameters based on different input shape and hardware for better performance
    constexpr uint32_t wg_tile_m = 256;
    constexpr uint32_t wg_tile_n = 256;
    constexpr uint32_t sg_tile_m = 32;
    constexpr uint32_t sg_tile_n = 64;

    //There are implicit requirement for sg_tile_k range
    constexpr uint32_t sg_tile_k = 32;

    // Org the compute shape for sub-matrix
    using tile_shape
            = xetla::group::tile_shape_t<wg_tile_n, // workgroup size in dim0
                    wg_tile_m, //	workgroup size in dim1
                    sg_tile_n, //	subgroup size in dim0
                    sg_tile_m>; //	subgroup size in dim1

    // Mirco-kernel configuration
    using brgemm_config = xetla::group::brgemm_selector_t<
            data_type_a, // input datatype for A
            data_type_b, // input datatype for B
            mem_layout::row_major, // memory layout for A
            mem_layout::row_major, // memory layout for B
            mem_space::global, // memory reading from global mem for A
            mem_space::global, // memory reading from global mem for B
            8, // buffer alignment for A, in unit of element
            8, // buffer alignment for B, in unit of element
            data_type_acc, // accumulator data type for intermediate resutls
            tile_shape, // computation tile shape
            sg_tile_k, // elements in each iteration
            mma_engine::xmx, // compute engine
            gpu_arch::Xe> // GPU arch
            ::brgemm;

    using epilogue_t = xetla::group::epilogue_t<
            xetla::group::epilogue_policy_default<gpu_arch::Xe>, tile_shape,
            mem_desc_t<data_type_c, mem_layout::row_major, mem_space::global>>;

    using gemm_op_t = xetla::kernel::gemm_t<
            gpu::xetla::kernel::dispatch_policy_default<gpu_arch::Xe>,
            brgemm_config, epilogue_t>;

    cl::sycl::nd_range<3> NDRange = gemm_op_t::get_nd_range(matrix_m, matrix_n);

    uint32_t warmup = 10;
    long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k;
    profiling_helper prof("basic_gemm", ops, "gflops");
    for (uint32_t i = 0; i < iter + warmup; i++) {
        if (i >= warmup) { prof.cpu_start(); }
        auto gpu_event = Queue.submit([&](handler &cgh) {
            // GPU kernel
            cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
                xetla_exec_item<3> ei(item);
                gemm_op_t gemm_op;
                // allocate slm and nbarrier resource
                slm_barrier_init<gemm_op_t>();
                // set up gemm arguments
                typename gemm_op_t::arguments_t arg(matrix_m, matrix_k,
                        matrix_n, A, matrix_k, B, matrix_n, C, matrix_n);
                gemm_op(ei, arg);
            });
        });
        gpu_event.wait();

        if (i >= warmup) {
            prof.cpu_end();
            prof.add_gpu_event(gpu_event);
        }
    }

    ASSERT_EQ(0,
            gemm_result_validate(A, B, C, matrix_m, matrix_k, matrix_n,
                    mem_layout::row_major, mem_layout::row_major));

    //performance
    prof.print_profiling_result(profiling_selector::GPU);

    free(A, Context);
    free(B, Context);
    free(C, Context);
}

int main() {
    basic_gemm_run(10);
    return (0);
}
