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

#include "common.hpp"
#include "habana_tests.hpp"
#include "kernel_func.hpp"
#include "utils/utils.hpp"
#include <gtest/gtest.h>

std::string esimd_compile_string
        = " -vc-codegen -doubleGRF "
          " -vc-disable-indvars-opt "
          " -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' ";

template <typename T>
class bf16_gemm_test : public ::testing::Test {};
TYPED_TEST_SUITE_P(bf16_gemm_test);
TYPED_TEST_P(bf16_gemm_test, esimd) {
    std::cout << "wg_swizzle_n_first" << std::endl;
    gemm_exec<TypeParam, result_validate<TypeParam>,
            bf16_gemm_func_default<TypeParam>>(esimd_compile_string);
    std::cout << "wg_swizzle_m_first" << std::endl;
    gemm_exec<TypeParam, result_validate<TypeParam>,
            bf16_gemm_func_m_first<TypeParam>>(esimd_compile_string);
}
REGISTER_TYPED_TEST_SUITE_P(bf16_gemm_test, esimd);
// using tests = ::testing::Types<Habana_Batch_Test0,Habana_Batch_Test1,Habana_Batch_Test2,Habana_Batch_Test3,Habana_Batch_Test4>;
using tests = ::testing::Types<Habana_Test0, Habana_Test1, Habana_Test2,
        Habana_Test3, Habana_Test4, Habana_Test5, Habana_Test6, Habana_Test7,
        Habana_Test8, Habana_Test9, Habana_Test10>;
// using tests = ::testing::Types<Test1, Test2, Test3, Test4, Test5, Test6, Test7, Test8, Test9, Test10,Test11,Test12,Test13,Test14,Test15>;
INSTANTIATE_TYPED_TEST_SUITE_P(bf16_gemm_test_suite, bf16_gemm_test, tests);
