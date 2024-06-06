# how to add a case
  # add a new case like " gemm_softmax<32, 12288, 32, 512, 32>(4, 4096, 12288)" in https://github.com/intel/xetla/blob/gemm_softmax_benchmark/examples/06_gemm_softmax/gemm_softmax.cpp#L361

# build and run
  # ./gemm_softmax_bf16_bf16_float_float.sh


export SYCL_PROGRAM_COMPILE_OPTIONS=" -vc-codegen -vc-disable-indvars-opt -doubleGRF -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' "

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
rm -rf $script_dir/build && mkdir $script_dir/build && cd $script_dir/build
source ./../tools/scripts/env.sh
cmake .. && make gemm_softmax && ./examples/06_gemm_softmax/gemm_softmax
