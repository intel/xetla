export SYCL_PROGRAM_COMPILE_OPTIONS=" -vc-codegen -vc-disable-indvars-opt -doubleGRF -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' "

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
repo_path=$script_dir/../../../
rm -rf $repo_path/build && mkdir $repo_path/build && cd $repo_path/build
source ../tools/scripts/env.sh
cmake .. && make gemm_relu && ./examples/12_gemm_relu/gemm_relu
