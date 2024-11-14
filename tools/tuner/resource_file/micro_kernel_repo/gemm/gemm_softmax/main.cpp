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
#include "test.hpp"
#include "tests/utils/utils.hpp"
#include "utils/utils.hpp"
#include "xetla.hpp"

using namespace gpu::xetla;
using namespace cl::sycl;

// flush cache 0: NO flush
// flush cache 1: memset
// flush cache 2: pingpong moving ptr offset
#define FLUSH_CACHE 0
#define WITHOUT_SOFTMAX

bool enable_validation = false;

template <class Test>
bool can_implement(typename Test::data_type_a *A, typename Test::data_type_b *B,
        typename Test::data_type_c *C, uint32_t mat_m, uint32_t mat_n,
        uint32_t mat_k, uint32_t lda, uint32_t ldb, uint32_t ldc,
        typename Test::data_type_bias *bias_ptr,
        typename Test::data_type_res *res_ptr0,
        typename Test::data_type_res *res_ptr1,
        typename Test::data_type_acc *acc_ptr, uint32_t *cnt_ptr) {
    if (Test::global_kslicing > 1) {
        std::cout << "global_kslicing = " << Test::global_kslicing
                  << " is not supported! Please modify the kernel!"
                  << std::endl;
        return false;
    }
    using gemm_functor = gemm_test_func<typename Test::data_type_a,
            typename Test::data_type_b, typename Test::data_type_c,
            typename Test::data_type_bias, typename Test::data_type_res,
            typename Test::data_type_acc, Test::wg_m, Test::wg_n, Test::sg_m,
            Test::sg_n, Test::sg_k, Test::layout_a, Test::layout_b,
            Test::global_kslicing, Test::local_kslicing, Test::fused_op>;

    return gemm_functor::can_implement(A, B, C, mat_m, mat_n, mat_k, lda, ldb,
            ldc, bias_ptr, res_ptr0, res_ptr1, acc_ptr, cnt_ptr);
}

template <typename data_type_a, typename data_type_b, typename data_type_c,
        typename data_type_acc = float>
int gemm_softmax_result_validate(data_type_a *A_device, data_type_b *B_device,
        data_type_c *C_device, uint32_t m, uint32_t k, uint32_t n,
        uint32_t batch_num, sycl::queue &queue,
        mem_layout mem_layout_a_ = mem_layout::row_major,
        mem_layout mem_layout_b_ = mem_layout::row_major) {
    uint32_t size_a = m * k;
    uint32_t size_b = k * n;
    uint32_t size_c = m * n;

    auto A_ptr = alloc_host_and_copy<data_type_a>(
            A_device, batch_num * size_a, queue);
    auto B_ptr = alloc_host_and_copy<data_type_b>(
            B_device, batch_num * size_b, queue);
    auto C_ptr = alloc_host_and_copy<data_type_c>(
            C_device, batch_num * size_c, queue);

    std::vector<data_type_acc> tmp_A(A_ptr, A_ptr + batch_num * size_a);
    std::vector<data_type_acc> tmp_B(B_ptr, B_ptr + batch_num * size_b);
    std::vector<data_type_acc> gold_C(batch_num * size_c, 0);
    for (uint32_t batch_id = 0; batch_id < batch_num; batch_id++) {
        get_gemm_gold(m, n, k, mem_layout_a_, mem_layout_b_,
                tmp_A.data() + batch_id * size_a,
                tmp_B.data() + batch_id * size_b,
                gold_C.data() + batch_id * size_c);
    }

#if !defined(WITHOUT_SOFTMAX)
    for (uint32_t batch_id = 0; batch_id < batch_num; ++batch_id) {
        for (uint32_t i = 0; i < m; i++) {
            data_type_acc row_max = 0;
            data_type_acc exp_sum = 0;
            uint32_t sfx_offset = batch_id * size_c + i * n;
            for (uint32_t j = 0; j < n; ++j) {
                row_max = max(row_max, gold_C[sfx_offset + j]);
            }
            for (uint32_t j = 0; j < n; ++j) {
                gold_C[sfx_offset + j]
                        = std::exp(gold_C[sfx_offset + j] - row_max);
                exp_sum += gold_C[sfx_offset + j];
            }
            for (uint32_t j = 0; j < n; ++j) {
                gold_C[sfx_offset + j] /= exp_sum;
            }
        }
    }
#endif

    buff_cmp::buff_vals<data_type_c> data(C_ptr, m * batch_num, n, n);
    buff_cmp::buff_vals<data_type_c, data_type_acc> other(
            gold_C.data(), m * batch_num, n, n);
    bool result
            = buff_cmp::xetla_buff_cmp(data, other, "gemm_softmax validation");

    free(A_ptr);
    free(B_ptr);
    free(C_ptr);

    std::cout << ((!result) ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}

template <class Test>
void gemm_run(int iter) {

    //Accept incoming parameters
    constexpr size_t matrix_m = Test::mat_m;
    constexpr size_t matrix_n = Test::mat_n;
    constexpr size_t matrix_k = Test::mat_k;
    constexpr size_t wg_tile_m = Test::wg_m;
    constexpr size_t wg_tile_n = Test::wg_n;
    constexpr size_t sg_tile_m = Test::sg_m;
    constexpr size_t sg_tile_n = Test::sg_n;
    constexpr size_t sg_tile_k = Test::sg_k;
    constexpr fused_type fused_op = Test::fused_op;
    //    constexpr size_t batch_num = Test::batch;
    constexpr size_t batch_num = 1;

    // default set Thread num = 32 to maximize EU utilization
    constexpr uint32_t thread_num = 32;

#if !defined(WITHOUT_SOFTMAX)
    assert(matrix_n == wg_tile_n);
#endif

    constexpr size_t size_a = matrix_m * matrix_k;
    constexpr size_t size_b = matrix_k * matrix_n;
    constexpr size_t size_c = matrix_m * matrix_n;
    constexpr uint32_t num_buffer = 1;

    using data_type_a = typename Test::data_type_a;
    using data_type_b = typename Test::data_type_b;
    using data_type_c = typename Test::data_type_c;
    using data_type_res = typename Test::data_type_res;
    using data_type_bias = typename Test::data_type_bias;
    using data_type_acc = typename Test::data_type_acc;
    using data_type_sfx = typename Test::data_type_acc;

    uint32_t warmup = 10;
    static constexpr auto l3_cache_size = 256 * 1024 * 1024;
    static constexpr auto pingpong_flush_iter = 3;

    //Turn on the enable_profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling()};

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

    // here keep the same dim in CM and esimd, diff the index in kernel code
    size_t group_range_m = (matrix_m % wg_tile_m == 0)
            ? matrix_m / wg_tile_m
            : (matrix_m / wg_tile_m) + 1;
    size_t group_range_n = (matrix_n % wg_tile_n == 0)
            ? matrix_n / wg_tile_n
            : (matrix_n / wg_tile_n) + 1;
    size_t subgroup_range_m = (wg_tile_m % sg_tile_m == 0)
            ? wg_tile_m / sg_tile_m
            : (wg_tile_m / sg_tile_m) + 1;
    size_t subgroup_range_n = (wg_tile_n % sg_tile_n == 0)
            ? wg_tile_n / sg_tile_n
            : (wg_tile_n / sg_tile_n) + 1;
    //    static_assert(subgroup_range_m * subgroup_range_n == thread_num,
    //             "Given thread number should equal to pre-set value 32!");

    std::cout << "MKNL: " << matrix_m << ", " << matrix_k << ", " << matrix_n
              << ", " << batch_num << ", Config: " << wg_tile_m << ", "
              << wg_tile_n << ", " << sg_tile_m << ", " << sg_tile_n << ", "
              << sg_tile_k << std::endl;

    std::cout << "group_num_x: " << group_range_n
              << ", group_num_y: " << group_range_m
              << ", group_num_z: " << Test::global_kslicing << "\n";
    std::cout << "group_size_x: " << subgroup_range_n
              << ", group_size_y: " << subgroup_range_m
              << ", group_size_z: " << Test::local_kslicing << std::endl;

    cl::sycl::range<3> group_range {batch_num, group_range_m, group_range_n};
    cl::sycl::range<3> local_range {1, subgroup_range_m, subgroup_range_n};
    cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);

    test_result result = test_result::complete;
    long ops_flo
            = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k * batch_num;

    long ops_hbm = (sizeof(data_type_a) * matrix_m * matrix_k
                           + sizeof(data_type_b) * matrix_k * matrix_n
                           + sizeof(data_type_c) * matrix_m * matrix_n)
            * batch_num;

    profiling_helper prof_flo("gemm_softmax", ops_flo, "gflops");
    profiling_helper prof_hbm("gemm_softmax", ops_hbm, "GB/s");

    try {
        for (uint32_t i = 0; i < iter + warmup; i++) {
#if FLUSH_CACHE == 1
            queue.memset((void *)(dev_cache), 0, l3_cache_size).wait();
#endif

            if (i >= warmup) {
                prof_flo.cpu_start();
                prof_hbm.cpu_start();
            }
            auto gpu_event = queue.submit([&](handler &cgh) {
                cgh.parallel_for<
                        Test>(nd_range, [=](nd_item<3> item) KERNEL_MAIN {
                    using namespace gpu::xetla;
                    using namespace gpu::xetla::group;
                    using namespace gpu::xetla::kernel;
                    using namespace gpu::xetla::subgroup;

                    uint32_t batch_id = item.get_group(0);

                    // Performance tuning setting based on different shapes
                    static constexpr uint32_t periodic_sync_interval = 8;
                    static constexpr uint32_t prefetch_distance = 3;
                    // should larger than 8
                    static constexpr uint32_t k_iter_num = sg_tile_k;

                    // Step 1: define mirco-kernel's configuration
                    using wg_shape = shape<wg_tile_n, wg_tile_m>;
                    using sg_shape = shape<sg_tile_n, sg_tile_m>;

                    // Mirco-kernel configuration
                    using tune_option = dict_t<
                            elem_v_t<tune_key::param_optimizer_type,
                                    tune_key_value::
                                            param_optimizer_decision_tree>,
                            elem_t_t<tune_key::sg_tile_shape, sg_shape>,
                            elem_v_t<tune_key::prefetch_distance,
                                    prefetch_distance>,
                            elem_v_t<tune_key::periodic_sync_interval,
                                    periodic_sync_interval>>;
                    using gemm_op_t = xetla::group::default_gemm_selector_t<
                            data_type_a, // input datatype for A
                            mem_layout::row_major, // memory layout for A
                            8, // leading dimension for A, in unit of element
                            mem_space::
                                    global, // memory reading from global mem for A
                            data_type_b, // input datatype for B
                            mem_layout::row_major, // memory layout for B
                            8, // leading dimension for B, in unit of element
                            mem_space::
                                    global, // memory reading from global mem for B
                            data_type_sfx, // accumulator data type for intermediate resutls
                            wg_shape, // computation tile shape
                            k_iter_num, // elements in each iteration
                            gpu_arch::Xe, // GPU arch
                            tune_option>;

                    using gemm_args_t = gemm_op_t::arguments_t;

                    using epilogue_t = xetla::group::default_epilogue_selector_t<
                            data_type_c, // onput datatype for C
                            mem_layout::row_major, // memory layout for C
                            8, // leading dimension for C, in unit of element
                            mem_space::
                                    global, // memory writing to global mem for C
                            wg_shape, // computation tile shape
                            k_iter_num, // elements in each iteration
                            gpu_arch::Xe, // GPU arch
                            tune_option>;

                    // using experimental::group::softmax
                    // define softmax forward op
                    using tile_shape = typename gemm_op_t::tile_shape;
#if !defined(WITHOUT_SOFTMAX)
                    using softmax_fwd_t = softmax_t<
                            softmax_policy_fwd<data_type_sfx, gpu_arch::Xe>,
                            tile_shape>;
                    using softmax_fwd_args_t =
                            typename softmax_fwd_t::arguments_t;

                    static constexpr uint32_t barrier_count
                            = gemm_op_t::barrier_count
                            + softmax_fwd_t::get_barrier_count::count;
                    static constexpr uint32_t slm_size = gemm_op_t::slm_size
                            + softmax_fwd_t::get_slm_size::size;
#else
                    // initialize shared local memory and named barrier
                    static constexpr uint32_t barrier_count  = gemm_op_t::barrier_count;
                    static constexpr uint32_t slm_size = gemm_op_t::slm_size;
#endif

                    xetla_nbarrier_init<barrier_count>();
                    xetla_local_init<slm_size>();

#if FLUSH_CACHE == 2
                    data_type_a *matA_ptr = A
                            + (i % pingpong_flush_iter) * pingpong_size_a
                            + batch_id * size_a;
                    uint32_t matA_ld = matrix_k;
                    data_type_b *matB_ptr = B
                            + (i % pingpong_flush_iter) * pingpong_size_b
                            + batch_id * size_b;
                    uint32_t matB_ld = matrix_n;
                    data_type_c *matC_ptr = C
                            + (i % pingpong_flush_iter) * pingpong_size_c
                            + batch_id * size_c;
#else
                    data_type_a *matA_ptr = A
                            + batch_id * size_a;
                    uint32_t matA_ld = matrix_k;
                    data_type_b *matB_ptr = B
                            + batch_id * size_b;
                    uint32_t matB_ld = matrix_n;
                    data_type_c *matC_ptr = C
                            + batch_id * size_c;
#endif

                    uint32_t matC_ld = matrix_n;

                    // ecah workgroup gets it individual index to start computation
                    int start_n = item.get_group(2) * wg_tile_n;
                    int start_m = item.get_group(1) * wg_tile_m;
                    int start_k = 0;
                    uint32_t wg_tile_k = matrix_k;
                    uint32_t boundary_n = (start_n + wg_tile_n) > matrix_n
                            ? matrix_n
                            : (start_n + wg_tile_n);
                    uint32_t boundary_m = (start_m + wg_tile_m) > matrix_m
                            ? matrix_m
                            : (start_m + wg_tile_m);
                    uint32_t boundary_k = wg_tile_k;
                    uint32_t inner_loop_count
                            = (wg_tile_k + k_iter_num - 1) / k_iter_num;

                    // initialize the memory description of matA & matB & matC
                    using mem_desc_a_t = typename gemm_op_t::mem_desc_a_t;
                    using mem_desc_b_t = typename gemm_op_t::mem_desc_b_t;
                    using mem_desc_c_t = typename epilogue_t::mem_desc_c_t;
                    mem_desc_a_t mem_desc_a(matA_ptr,
                            {boundary_k, boundary_m, matA_ld},
                            {start_k, start_m});
                    mem_desc_b_t mem_desc_b(matB_ptr,
                            {boundary_n, boundary_k, matB_ld},
                            {start_n, start_k});
                    mem_desc_c_t mem_desc_c(matC_ptr,
                            {boundary_n, boundary_m, matC_ld},
                            {start_n, start_m});

                    // call gemm function and result will be written in matAcc
                    using gemm_args_t = typename gemm_op_t::arguments_t;
                    gemm_args_t gemm_args(
                            mem_desc_a, mem_desc_b, inner_loop_count);

                    typename gemm_op_t::work_group_t g(
                            item.get_local_linear_id());
                    typename gemm_op_t::matAcc_t matAcc(0);

                    gemm_op_t gemm;
                    gemm(g, matAcc, gemm_args);
#if !defined(WITHOUT_SOFTMAX)
                    // for each matAcc, call softmax op
                    softmax_fwd_t softmax_fwd;
                    softmax_fwd_args_t softmax_fwd_args(1);
                    softmax_fwd(g, matAcc, {}, softmax_fwd_args);
#endif

                    // write matAcc value into pointer C
                    epilogue_t epilogue;
                    epilogue(g, matAcc, mem_desc_c);
                });
            });
            gpu_event.wait();

            if (i >= warmup) {
                prof_flo.cpu_end();
                prof_hbm.cpu_end();
                prof_flo.add_gpu_event(gpu_event);
                prof_hbm.add_gpu_event(gpu_event);
            }
        }
    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        FAIL();
    }

    int err_cnt = gemm_softmax_result_validate<data_type_a, data_type_b,
            data_type_c, data_type_acc>(A, B, C, matrix_m, matrix_k, matrix_n,
            batch_num, queue, mem_layout::row_major, mem_layout::row_major);
    ASSERT_EQ(0, err_cnt);

    // performance
    //    prof_flo.print_profiling_result(profiling_selector::GPU);
    prof_hbm.print_profiling_result(profiling_selector::GPU);

    free(A, context);
    free(B, context);
    free(C, context);
#if FLUSH_CACHE == 1
    free(dev_cache, context);
#endif

    if (result == test_result::skip) {
        GTEST_SKIP();
    } else if (result != test_result::complete) {
        FAIL();
    }
}

template <typename T>
class tuner_kernel_gemm_softmax : public ::testing::Test {};
TYPED_TEST_SUITE_P(tuner_kernel_gemm_softmax);
TYPED_TEST_P(tuner_kernel_gemm_softmax, esimd) {
    gemm_run<TypeParam>(ITER);
}

REGISTER_TYPED_TEST_SUITE_P(tuner_kernel_gemm_softmax, esimd);
INSTANTIATE_TYPED_TEST_SUITE_P(
        tuner_kernel_gemm_softmax_test_suite, tuner_kernel_gemm_softmax, tests);

int main(int argc, char **argv) {
    if (argc > 1) {
        string arg = argv[1];
        if (arg == "1") { enable_validation = true; }
    }
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
