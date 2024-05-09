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

#pragma once

#include "common.hpp"
#include "profiling.hpp"
#include "xetla.hpp"

using namespace cl::sycl;
using namespace gpu;
using namespace gpu::xetla;

#define CACHE_FLUSH 1

inline size_t time_event(sycl::event &e) {
    // get start and end times
    cl_ulong start_time = e.template get_profiling_info<
            sycl::info::event_profiling::command_start>();

    cl_ulong end_time = e.template get_profiling_info<
            sycl::info::event_profiling::command_end>();

    // return the delta
    return static_cast<size_t>(end_time - start_time);
}

template <typename data_type>
inline data_type *alloc_device_and_init(size_t size, size_t test_iter,
        std::function<void(data_type *data, size_t elements)> init_func,
        sycl::queue &queue, sycl::device &device, sycl::context &context) {
    auto host_ptr = static_cast<data_type *>(malloc(size * sizeof(data_type)));

    for (size_t i = 0; i < size; ++i) {
        init_func(host_ptr, i);
    }

    auto device_ptr = static_cast<data_type *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    test_iter * size * sizeof(data_type), device, context));
    for (int it = 0; it < test_iter; it++) {
        queue.memcpy((void *)(device_ptr + it * size), (void *)host_ptr,
                     size * sizeof(data_type))
                .wait();
    }

    free(host_ptr);

    return device_ptr;
}

template <class Test, typename validate_func, typename KERNEL,
        int SLMSIZE = 128 * 1024, int BARNUM = 32>
void gemm_exec(const std::string &compile_str, size_t batch = 1) {
    test_result result = test_result::complete;

    using gemm_op_t = typename KERNEL::gemm_op_t;

    using data_type_a = typename Test::data_type_a;
    using data_type_b = typename Test::data_type_b;
    using data_type_c = typename Test::data_type_c;
    using data_type_acc = typename Test::data_type_acc;

    int iter = 10, warmup = 10;
#ifdef CACHE_FLUSH
    batch = iter + warmup;
#else
    batch = 1;
#endif
    constexpr size_t matrix_m = Test::mat_m;
    constexpr size_t matrix_n = Test::mat_n;
    constexpr size_t matrix_k = Test::mat_k;

    size_t size_a = matrix_m * matrix_k;
    size_t size_b = matrix_k * matrix_n;
    size_t size_c = matrix_m * matrix_n;
    sycl::property_list properties {sycl::property::queue::enable_profiling()};
    auto queue = sycl::queue(properties);
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Problem size MKN:" << matrix_m << "x" << matrix_k << "x"
              << matrix_n << std::endl;
    std::cout << "Running on test iter: " << batch << ", "
              << device.get_info<info::device::name>() << "\n";

    auto A = alloc_device_and_init<data_type_a>(
            size_a, batch,
            [](data_type_a *data, size_t idx) {
                data[idx] = static_cast<data_type_a>(random_float());
            },
            queue, device, context);
    auto B = alloc_device_and_init<data_type_b>(
            size_b, batch,
            [](data_type_b *data, size_t idx) {
                data[idx] = static_cast<data_type_b>(random_float());
            },
            queue, device, context);
    auto C = alloc_device_and_init<data_type_c>(
            size_c, batch,
            [](data_type_c *data, size_t idx) {
                data[idx] = static_cast<data_type_c>(0);
            },
            queue, device, context);

    size_t size_acc = gemm_op_t::get_acc_buf_size(matrix_m, matrix_n);
    size_t size_cnt = gemm_op_t::get_cnt_buf_size(matrix_m, matrix_n);
    auto Acc = alloc_device_and_init<data_type_acc>(
            size_acc, batch,
            [](data_type_acc *data, size_t idx) {
                data[idx] = static_cast<data_type_acc>(0);
            },
            queue, device, context);
    auto Cnt = alloc_device_and_init<uint32_t>(
            size_cnt, batch,
            [](uint32_t *data, size_t idx) {
                data[idx] = static_cast<uint32_t>(0);
            },
            queue, device, context);

    try {
        std::vector<kernel_id> kernelId = {get_kernel_id<Test>()};
        auto inputBundle
                = get_kernel_bundle<bundle_state::input>(context, kernelId);
        setenv("SYCL_PROGRAM_COMPILE_OPTIONS", compile_str.c_str(), 1);
        kernel_bundle<bundle_state::executable> exeBundle = build(inputBundle);
        unsetenv("SYCL_PROGRAM_COMPILE_OPTIONS");

        using namespace gpu::xetla::group;
        using namespace gpu::xetla::kernel;
        using namespace gpu::xetla::subgroup;

        typename gemm_op_t::arguments_t arg(matrix_m, matrix_k, matrix_n,
                nullptr,
                Test::layout_a == mem_layout::col_major ? matrix_m : matrix_k,
                nullptr,
                Test::layout_b == mem_layout::col_major ? matrix_k : matrix_n,
                nullptr, matrix_n, nullptr, nullptr);
        arg.matA_base = A;
        arg.matB_base = B;
        arg.matC_base = C;
        arg.acc_base = Acc;
        arg.cnt_base = Cnt;
        // if (!gemm_op_t::can_implement(arg)) {
        //     std::cout << "The arguments cannot be supported, skip ... "
        //               << std::endl;
        //     result = test_result::skip;
        // }
        cl::sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(arg);

        std::vector<float> event_times(iter + warmup);
        for (uint32_t j = 0; j < iter + warmup; j++) {

            auto e_esimd = queue.submit([&](handler &cgh) {
                cgh.use_kernel_bundle(exeBundle);
                cgh.parallel_for<Test>(
                        nd_range, [=](nd_item<3> item) KERNEL_MAIN {
                // int batch_idx = item.get_workgroup(0);
#ifdef CACHE_FLUSH
                            int batch_idx = j;
                            auto A_ptr = A + batch_idx * size_a;
                            auto B_ptr = B + batch_idx * size_b;
                            auto C_ptr = C + batch_idx * size_c;
                            auto Acc_ptr = Acc + batch_idx * size_acc;
                            auto Cnt_ptr = Cnt + batch_idx * size_cnt;
#else
                            auto A_ptr = A;
                            auto B_ptr = B;
                            auto C_ptr = C;
                            auto Acc_ptr = Acc;
                            auto Cnt_ptr = Cnt;
#endif
                            gpu::xetla::xetla_local_init<SLMSIZE>();
                            gpu::xetla::xetla_nbarrier_init<BARNUM>();
                            KERNEL::run(item, A_ptr, B_ptr, C_ptr, matrix_m,
                                    matrix_n, matrix_k, Acc_ptr, Cnt_ptr);
                        });
            });

            e_esimd.wait();
            event_times[j] = time_event(e_esimd) / 1e9;
        }
        double average_event_time = 0.f;
        auto best = 999.f;
        for (uint32_t i = warmup; i < iter + warmup; i++) {
            printf("GPU time is %f ms, Tflops is: %f, HBM (GBs) is %f\n",
                    event_times[i] * 1e3,
                    2.0 * matrix_m * matrix_n * matrix_k / 1e12
                            / event_times[i],
                    (matrix_m * matrix_k * sizeof(data_type_a)
                            + matrix_k * matrix_n * sizeof(data_type_b)
                            + matrix_m * matrix_n * sizeof(data_type_c))
                            / event_times[i] / 1e9);
            average_event_time += event_times[i];
            best = min(best, event_times[i]);
        }
        average_event_time /= iter;
        printf("Best is %f Tflops, %f HBM (GBs)\n",
                2.0 * matrix_m * matrix_n * matrix_k / 1e12 / best,
                (matrix_m * matrix_k * sizeof(data_type_a)
                        + matrix_k * matrix_n * sizeof(data_type_b)
                        + matrix_m * matrix_n * sizeof(data_type_c))
                        / best / 1e9);

    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        result = test_result::fail;
    }

    // validation
    if (result == test_result::complete) {
        validate_func vfunc;
        ASSERT_EQ(0, vfunc(A, B, C, queue));
    }

    free(A, context);
    free(B, context);
    free(C, context);
    free(Acc, context);
    free(Cnt, context);

    if (result == test_result::skip) {
        GTEST_SKIP();
    } else if (result != test_result::complete) {
        FAIL();
    }
}

/// @brief The template function to execute kernel in esimd way for unit test framework
///
/// @tparam data_type data_type The data type of buffer used in kernel and buffer allocation
/// @tparam KERNEL the kernel function struct
/// @param nd_range the range of workitems
/// @param validate_result validation function, taking 3 parameters buffer A, B as input C as output
///
template <typename data_type, class KERNEL, size_t SLMSIZE = 128 * 1024,
        size_t BARNUM = 32, size_t Size = 4096>
void kernel_run(auto nd_range, auto validate_result) {

    queue queue {};
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();
    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

    auto A = alloc_device_and_init<data_type>(
            Size,
            [](data_type *data, size_t idx) {
                data[idx] = static_cast<data_type>(idx);
            },
            queue, device, context);
    auto B = alloc_device_and_init<data_type>(
            Size,
            [](data_type *data, size_t idx) {
                data[idx] = static_cast<data_type>(idx);
            },
            queue, device, context);
    auto C = alloc_device_and_init<data_type>(
            Size,
            [](data_type *data, size_t idx) {
                data[idx] = static_cast<data_type>(idx);
            },
            queue, device, context);

    try {
        auto e_esimd = queue.submit([&](handler &cgh) {
            cgh.parallel_for<>(nd_range, [=](nd_item<1> ndi) KERNEL_MAIN {
                gpu::xetla::xetla_local_init<SLMSIZE>();
                gpu::xetla::xetla_nbarrier_init<BARNUM>();
                KERNEL::run(&ndi, A, B, C);
            });
        });
        e_esimd.wait();
    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        FAIL();
    }

    auto A_host = alloc_host_and_copy<data_type>(A, Size, queue);
    auto B_host = alloc_host_and_copy<data_type>(B, Size, queue);
    auto C_host = alloc_host_and_copy<data_type>(C, Size, queue);

    ASSERT_EQ(0, validate_result(A_host, B_host, C_host));

    free(A, context);
    free(B, context);
    free(C, context);

    free(A_host);
    free(B_host);
    free(C_host);
}
