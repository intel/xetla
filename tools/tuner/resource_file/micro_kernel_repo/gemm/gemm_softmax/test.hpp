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

#include "kernel_func.hpp"
#include "utils/utils.hpp"
#include "xetla.hpp"
#include <gtest/gtest.h>

using namespace gpu::xetla;
using namespace cl::sycl;
//The number of times the kernel is executed
constexpr int ITER = 100;

class TestBase {
public:
    static constexpr size_t mat_m = 1024;
    static constexpr size_t mat_k = 64;
    static constexpr size_t mat_n = 1024;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 1024;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_k = 32;
    static constexpr size_t sg_n = 64;

    static constexpr uint32_t global_kslicing = 1;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    static constexpr fused_type fused_op = fused_type::none;

    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = bf16;
    using data_type_bias = bf16;
    using data_type_res = bf16;
    using data_type_acc = float;

    template <class T>
    static std::string name() {
        std::string mem_layout_a_str
                = T::layout_a == mem_layout::col_major ? "T" : "N";
        std::string mem_layout_b_str
                = T::layout_b == mem_layout::col_major ? "T" : "N";
        std::string name = std::to_string(T::mat_m) + "x"
                + std::to_string(T::mat_k) + "x" + std::to_string(T::mat_n)
                + "_" + std::to_string(T::wg_m) + "x" + std::to_string(T::wg_n)
                + "_" + std::to_string(T::sg_m) + "x" + std::to_string(T::sg_k)
                + "x" + std::to_string(T::sg_n) + "_" + mem_layout_a_str
                + mem_layout_b_str + "_"
                + getTypeName<typename T::data_type_a>() + "_"
                + getTypeName<typename T::data_type_b>() + "_"
                + getTypeName<typename T::data_type_c>() + "_l3k"
                + std::to_string(T::global_kslicing) + "_slmk"
                + std::to_string(T::local_kslicing);
        return name;
    }
};

class gemm_softmax_shape : public TestBase {
public:
    static constexpr uint32_t batch_size = 1;
    static constexpr size_t mat_m = 1024;
    static constexpr size_t mat_k = 64;
    static constexpr size_t mat_n = 1024;
    static constexpr size_t wg_m = 64;
    static constexpr size_t wg_n = 1024;
    static constexpr size_t sg_m = 64;
    static constexpr size_t sg_k = 4;
    static constexpr size_t sg_n = 32;
    static constexpr uint32_t global_kslicing = 1;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = bf16;
};

using tests = ::testing::Types<gemm_softmax_shape>;
