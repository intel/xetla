# Build And Run Benchmark

## GEMM Benchmark
### Build And Run
    $cd {XETLA_REPO}
    $./tools/scripts/benchmark/build_run_gemm.sh

    The output show the average/max/min Bandwidth and Tflops like:
        Tflops  [min: 199.014415, max: 303.908466, average: 297.487233]
        HBM(GBs)[min: 461.581285, max: 704.865826, average: 689.972831]

### Add a New Matrix Shape
- Add a new shape class like "class Test_xxx : public TestBase ..." in [tests/integration/gemm/bf16/common.hpp](./tests/integration/gemm/bf16/common.hpp).
- Add the Test "Test_xxx" to gtest list in [tests/integration/gemm/bf16/main.cpp](./tests/integration/gemm/bf16/main.cpp).



## Softmax Benchmark
### Build And Run
    $cd {XETLA_REPO}
    $./tools/scripts/benchmark/build_run_softmax.sh

    The output show the average/max/min Bandwidth like:
        average Bandwidth: GB/S: 1252.778991,max Bandwidth: GB/S: 1278.751180, min Bandwidth: GB/S: 1219.274418

### Add a New Matrix Shape
- Add a test case like "mat1_4096x256_bf16_xxx" in [tests/integration/softmax/softmax_config.hpp](./tests/integration/softmax/softmax_config.hpp).
- Add the test "mat1_4096x256_bf16_xxx" like "softmax_fwd_run<mat1_4096x256_bf16_xxx>();" in [tests/integration/softmax/softmax_fwd.cpp](./tests/integration/softmax/softmax_fwd.cpp).



## GEMM + Softmax Benchmark
### Build And Run
    $cd {XETLA_REPO}
    $./tools/scripts/benchmark/build_run_gemm_softmax.sh

    The output show the average/max/min Bandwidth like:
        ============== [kernel time] gflops   ==================
        [kernel time]The minimum gflops(GPU_time) is 22218
        [kernel time]The maximum gflops(GPU_time) is 29418.6
        [kernel time]The median  gflops(GPU_time) is 25499.3
        [kernel time]The mean    gflops(GPU_time) is 25441.1
        ======================================================

        ============== [kernel time] GB/s   ==================
        [kernel time]The minimum GB/s(GPU_time) is 21.6973
        [kernel time]The maximum GB/s(GPU_time) is 28.7291
        [kernel time]The median  GB/s(GPU_time) is 24.9016
        [kernel time]The mean    GB/s(GPU_time) is 24.8448

### Add a New Matrix Shape
- Add a new shape MKN 4x4096x12288 and wg_tile_m 32, wg_tile_n 12288, sg_tile_m 32, sg_tile_n 512, sg_tile_k 32, like "gemm_softmax<32, 12288, 32, 512, 32>(4, 4096, 12288)" in [examples/06_gemm_softmax/gemm_softmax.cpp](./examples/06_gemm_softmax/gemm_softmax.cpp).


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

