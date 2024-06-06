# how to add a case
  # 1. add a new test case "class Test_xxx : public TestBase ... " in tests/integration/gemm/bf16/common.hpp
  # 2. add the test name "Test_xxx" to tests/integration/gemm/bf16/main.cp

# build and run
  # ./gemm_benchmark_bf16_bf16_float_float.sh

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
rm -rf $script_dir/build && mkdir $script_dir/build && cd $script_dir/build
source ./../tools/scripts/env.sh
cmake .. && make gemm_bf16 && ./tests/integration/gemm/bf16/gemm_bf16
