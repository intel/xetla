# For shape < 2K, could enable doubleGRF
export SYCL_PROGRAM_COMPILE_OPTIONS=" -vc-codegen -vc-disable-indvars-opt -doubleGRF -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' "
# For shape > 2K, could not enable doubleGRF
#export SYCL_PROGRAM_COMPILE_OPTIONS=" -vc-codegen -vc-disable-indvars-opt -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' "

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
repo_path=${script_dir}/../../../
rm -rf $repo_path/build && mkdir $repo_path/build && cd $repo_path/build
source ../tools/scripts/env.sh
cmake .. && make gemm_bf16 && ./tests/integration/gemm/bf16/gemm_bf16
