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

#include <unistd.h>
#include "common.hpp"
#include "softmax_config.hpp"
#include "softmax_fwd_kernel.hpp"
#include "xetla.hpp"

using namespace gpu::xetla;

// flush cache 0: NO flush
// flush cache 1: memset
// flush cache 2: pingpong moving ptr offset
#define FLUSH_CACHE 2

template <class Test>
class FlushCache;

#if FLUSH_CACHE == 1
// flush cache 1: memset
void flush_cache_1(sycl::queue &queue, unsigned *ptr, size_t num_elem) {
    queue.submit([&](sycl::handler &cgh) {
             cgh.parallel_for(sycl::nd_range<1>(num_elem, 1),
                     [=](sycl::nd_item<1> item) {
                         auto linear_id = item.get_global_linear_id();
                         ptr[linear_id] = 0;
                     });
         }).wait();
}
#endif

//Test: accept different test data
//iter: indicate the iterations of the kernel
template <class Test>
void softmax_fwd_run() {
    //Accept incoming parameters
    size_t mat_n = Test::mat_n;
    size_t mat_m = Test::mat_m;
    static constexpr size_t sg_n = Test::sg_n;
    static constexpr size_t sg_m = Test::sg_m;
    static constexpr size_t wg_n = Test::wg_n;
    static constexpr size_t wg_m = Test::wg_m;

    using data_type_in = typename Test::data_type_in;
    using data_type_acc = typename Test::data_type_acc;
    using data_type_out = typename Test::data_type_out;

    int size_in = mat_n * mat_m;
    int size_out = mat_n * mat_m;

    //Turn on the enable_profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling()};
    auto queue = sycl::queue(properties);
    auto context = queue.get_info<sycl::info::queue::context>();
    auto device = queue.get_info<sycl::info::queue::device>();

    std::cout << "Running on " << device.get_info<sycl::info::device::name>()
              << "\n";

    constexpr int warm_up = 10;
    constexpr int rep = 10;
    constexpr int total_cnt = warm_up + rep;

#if FLUSH_CACHE == 1
    // Malloc the cache flush buffer.
    auto dev_cache = alloc_device_and_init<unsigned>(
            256 * 1024 * 1024 / 4, // 256M bytes
            [](unsigned *data, size_t idx) {
                data[idx] = static_cast<unsigned>(0);
            },
            queue, device, context);
#endif

#if FLUSH_CACHE == 2
    static constexpr size_t l3_cache_size = 256 * 1024 * 1024;
    auto pingpong_size_in
            = std::max(l3_cache_size / sizeof(data_type_in), (size_t)size_in);
    auto pingpong_size_out
            = std::max(l3_cache_size / sizeof(data_type_out), (size_t)size_out);
    static constexpr auto pingpong_iter = 3;

    //Define and initialize the data required for the calculation
    auto buffer_in = alloc_device_and_init<data_type_in>(
            pingpong_size_in * pingpong_iter,
            [](data_type_in *data, size_t idx) {
                data[idx] = static_cast<data_type_in>(random_float());
            },
            queue, device, context);
    auto buffer_out = alloc_device_and_init<data_type_out>(
            pingpong_size_out * pingpong_iter,
            [](data_type_out *data, size_t idx) {
                data[idx] = static_cast<data_type_out>(0);
            },
            queue, device, context);
#else
    auto buffer_in = alloc_device_and_init<data_type_in>(
            size_in,
            [](data_type_in *data, size_t idx) {
                data[idx] = static_cast<data_type_in>(random_float());
            },
            queue, device, context);
    auto buffer_out = alloc_device_and_init<data_type_out>(
            size_out,
            [](data_type_out *data, size_t idx) {
                data[idx] = static_cast<data_type_out>(0);
            },
            queue, device, context);
#endif

    data_type_acc sqrt_dk_inv = 0.125f;

    size_t group_range_m = (mat_m + wg_m - 1) / wg_m;
    size_t group_range_n = (mat_n + wg_n - 1) / wg_n;
    size_t subgroup_range_m = (wg_m + sg_m - 1) / sg_m;
    size_t subgroup_range_n = (wg_n + sg_n - 1) / sg_n;

    std::cout << mat_m << ", " << mat_n << ", " << wg_m << ", " << wg_n << ", "
              << sg_m << ", " << sg_n << std::endl;

    cl::sycl::range<3> group_range {1, group_range_m, group_range_n};
    cl::sycl::range<3> local_range {1, subgroup_range_m, subgroup_range_n};
    cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);

    // esimd kernel prepratation and execution
    {
        std::vector<double> kernel_times;
        try {
            for (size_t i = 0; i < total_cnt; i++) {
#if FLUSH_CACHE == 1
                flush_cache_1(queue, dev_cache, 256 * 1024 * 1024 / 4);
                queue.wait_and_throw();
                usleep(300);
#endif

#if FLUSH_CACHE == 2
                auto ptr_in
                        = buffer_in + (i % pingpong_iter) * pingpong_size_in;
                auto ptr_out
                        = buffer_out + (i % pingpong_iter) * pingpong_size_out;
#else
                auto ptr_in = buffer_in;
                auto ptr_out = buffer_out;
#endif

                // kernel
                auto e_softmax_fwd = queue.submit([&](sycl::handler &cgh) {
                    // cgh.use_kernel_bundle(exeBundle);
                    cgh.parallel_for<Test>(
                            nd_range, [=](sycl::nd_item<3> item) KERNEL_MAIN {
                                using softmax_fwd_func
                                        = softmax_fwd_test_func<data_type_in,
                                                data_type_out, data_type_acc,
                                                wg_n, wg_m, sg_n, sg_m>;
                                constexpr uint32_t barrier_count
                                        = softmax_fwd_func::barrier_count;
                                constexpr uint32_t slm_size
                                        = softmax_fwd_func::slm_size;

                                xetla_nbarrier_init<barrier_count>();
                                xetla_local_init<slm_size>();

                                softmax_fwd_func::run(item, ptr_in, ptr_out,
                                        mat_m, mat_n, mat_n, sqrt_dk_inv);
                            });
                });
                e_softmax_fwd.wait();
                if (i < warm_up) continue;
                double time
                        = (e_softmax_fwd.template get_profiling_info<
                                   sycl::info::event_profiling::command_end>()
                                  - e_softmax_fwd.template get_profiling_info<
                                          sycl::info::event_profiling::
                                                  command_start>())
                        / (1000.0f * 1000.0f * 1000.f);

                kernel_times.push_back(time);
            }

        } catch (cl::sycl::exception const &e) {
            std::cout << "SYCL exception caught: " << e.what() << '\n';
            abort();
        }

        for (size_t i = 0; i < rep; i++) {
            double time = kernel_times[i];
            printf("M: %d, N: %d Data_type_in: %d, Bandwidth: GB/S: %f \n",
                    mat_m, mat_n, sizeof(data_type_in),
                    ((mat_m * mat_n * sizeof(data_type_in) * 2 / 1e9) / time));
        }
        double max = *max_element(kernel_times.begin(), kernel_times.end());
        double min = *min_element(kernel_times.begin(), kernel_times.end());
        double avg_time
                = std::accumulate(kernel_times.begin(), kernel_times.end(), 0.0)
                / kernel_times.size();
        printf("M: %d, N: %d Data_type_in: %d, average Bandwidth: GB/S: %f,"
               "max Bandwidth: GB/S: %f, "
               "min Bandwidth: GB/S: %f \n",
                mat_m, mat_n, sizeof(data_type_in),
                ((mat_m * mat_n * sizeof(data_type_in) * 2 / 1e9) / avg_time),
                ((mat_m * mat_n * sizeof(data_type_in) * 2 / 1e9) / min),
                ((mat_m * mat_n * sizeof(data_type_in) * 2 / 1e9) / max));
    }

    // validation
    auto buffer_in_host
            = alloc_host_and_copy<data_type_in>(buffer_in, size_in, queue);
    auto buffer_out_host
            = alloc_host_and_copy<data_type_out>(buffer_out, size_out, queue);
    int correct = fwd_reduction_result_validate<data_type_in, data_type_out,
            data_type_acc>(
            buffer_in_host, buffer_out_host, mat_m, mat_n, sqrt_dk_inv);

    assert(0 == correct);

#if FLUSH_CACHE == 1
    free(dev_cache, context);
#endif

    free(buffer_in, context);
    free(buffer_out, context);

    free(buffer_in_host);
    free(buffer_out_host);
}

int main() {
    //softmax_fwd_run<mat1_4096x256_bf16_cfg0>();
    softmax_fwd_run<mat1_4096x1024_bf16_cfg0>();
    softmax_fwd_run<mat1_4096x2048_bf16_cfg0>();
    softmax_fwd_run<mat1_4096x4096_bf16_cfg0>();
    softmax_fwd_run<mat1_4096x8192_bf16_cfg1>();
    // softmax_fwd_run<mat1_4096x4096_bf16_cfg2>();
    //  softmax_fwd_run<mat1_4096x4096_bf16_cfg3>();

    return 0;
}
