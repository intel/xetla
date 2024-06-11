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
#include "kernel_func.hpp"
#include "utils/buff_compare.hpp"
#include "utils/common.hpp"
#include <gtest/gtest.h>

class Habana_Test0 : public TestBase {
public:
    static constexpr size_t mat_m = 512;
    static constexpr size_t mat_n = 8192;
    static constexpr size_t mat_k = 8192;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t global_kslicing = 1;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Habana_Test1 : public TestBase {
public:
    static constexpr size_t mat_m = 512;
    static constexpr size_t mat_k = 8192;
    static constexpr size_t mat_n = 32768;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t global_kslicing = 1;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Habana_Test2 : public TestBase {
public:
    static constexpr size_t mat_m = 512;
    static constexpr size_t mat_k = 32768;
    static constexpr size_t mat_n = 8192;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t global_kslicing = 1;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Habana_Test3 : public TestBase {
public:
    static constexpr size_t mat_m = 16384;
    static constexpr size_t mat_k = 8192;
    static constexpr size_t mat_n = 1024;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t global_kslicing = 1;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Habana_Test4 : public TestBase {
public:
    static constexpr size_t mat_m = 16384;
    static constexpr size_t mat_k = 1024;
    static constexpr size_t mat_n = 8192;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t global_kslicing = 1;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Habana_Test5 : public TestBase {
public:
    static constexpr size_t mat_m = 16384;
    static constexpr size_t mat_k = 8192;
    static constexpr size_t mat_n = 4096;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t global_kslicing = 1;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Habana_Test6 : public TestBase {
public:
    static constexpr size_t mat_m = 16384;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t mat_n = 8192;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t global_kslicing = 1;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Habana_Test7 : public TestBase {
public:
    static constexpr size_t mat_m = 4096;
    static constexpr size_t mat_k = 16384;
    static constexpr size_t mat_n = 8192;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr uint32_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Habana_Test8 : public TestBase {
public:
    static constexpr size_t mat_m = 8192;
    static constexpr size_t mat_k = 16384;
    static constexpr size_t mat_n = 4096;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr uint32_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Habana_Test9 : public TestBase {
public:
    static constexpr size_t mat_m = 1024;
    static constexpr size_t mat_k = 16384;
    static constexpr size_t mat_n = 8192;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr uint32_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Habana_Test10 : public TestBase {
public:
    static constexpr size_t mat_m = 8192;
    static constexpr size_t mat_k = 16384;
    static constexpr size_t mat_n = 1024;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr uint32_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Habana_Batch_Test0 : public TestBase {
public:
    static constexpr size_t batch = 4096;
    static constexpr size_t mat_m = 8;
    static constexpr size_t mat_k = 128;
    static constexpr size_t mat_n = 16384;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 512;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 16;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr uint32_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Habana_Batch_Test1 : public TestBase {
public:
    static constexpr size_t batch = 4096;
    static constexpr size_t mat_m = 8;
    static constexpr size_t mat_k = 16384;
    static constexpr size_t mat_n = 128;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 512;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr uint32_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Habana_Batch_Test2 : public TestBase {
public:
    static constexpr size_t batch = 4;
    static constexpr size_t mat_m = 32768;
    static constexpr size_t mat_k = 128;
    static constexpr size_t mat_n = 4096;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr uint32_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Habana_Batch_Test3 : public TestBase {
public:
    static constexpr size_t batch = 4;
    static constexpr size_t mat_m = 32768;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t mat_n = 128;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr uint32_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = float;
    using data_type_acc = float;
};

class Habana_Batch_Test4 : public TestBase {
public:
    static constexpr size_t mat_m = 4096;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t mat_n = 128;
    static constexpr size_t wg_m = 64;
    static constexpr size_t wg_n = 128;
    static constexpr size_t sg_m = 16;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 32;
    static constexpr uint32_t local_kslicing = 1;
    static constexpr uint32_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = float;
    using data_type_acc = float;
};

template <class Test>
using bf16_gemm_func_default = bf16_gemm_test_func<typename Test::data_type_a,
        typename Test::data_type_b, typename Test::data_type_c,
        typename Test::data_type_acc,
        gpu::xetla::kernel::group_swizzle_default<gpu_arch::Xe>, Test::wg_m,
        Test::wg_n, Test::sg_m, Test::sg_n, Test::sg_k, Test::layout_a,
        Test::layout_b, Test::global_kslicing, Test::local_kslicing,
        Test::engine>;

template <class Test>
using bf16_gemm_func_m_first = bf16_gemm_test_func<typename Test::data_type_a,
        typename Test::data_type_b, typename Test::data_type_c,
        typename Test::data_type_acc,
        gpu::xetla::kernel::group_swizzle_m_first<Test::mat_m / Test::wg_m,
                gpu_arch::Xe>,
        Test::wg_m, Test::wg_n, Test::sg_m, Test::sg_n, Test::sg_k,
        Test::layout_a, Test::layout_b, Test::global_kslicing,
        Test::local_kslicing, Test::engine>;