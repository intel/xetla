# Intel® Xe Templates for Linear Algebra

## How to build and run gemm benchmark

### How to add a matrix shape
- add a new shape class like "class Test_xxx : public TestBase ..." in tests/integration/gemm/bf16/common.hpp
- add the Test "Test_xxx" to gtest list in tests/integration/gemm/bf16/main.cpp

### How to build and run
    tools/scripts/benchmark/build_run_gemm.sh

## How to build and run gemm + softmax benchmark
### How to add a matrix shape
- add a new shape MKN 4x4096x12288 and wg_tile_m 32, wg_tile_n 12288, sg_tile_m 32, sg_tile_n 512, sg_tile_k 32, like "gemm_softmax<32, 12288, 32, 512, 32>(4, 4096, 12288)" in examples/06_gemm_softmax/gemm_softmax.cpp

### How to build and run
    tools/scripts/benchmark/build_run_gemm_softmax.sh

## How to build and run softmax benchmark
### How to add a matrix shape
- add a test case like "mat1_4096x256_bf16_xxx" in tests/integration/softmax/softmax_config.hpp
- add the test "mat1_4096x256_bf16_xxx" like "softmax_fwd_run<mat1_4096x256_bf16_xxx>();" in tests/integration/softmax/softmax_fwd.cpp

### How to build and run
    tools/scripts/benchmark/build_run_softmax.sh

## Copyright

Copyright (c) 2022-2023 Intel Corporation
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

