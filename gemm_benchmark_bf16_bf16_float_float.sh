script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
rm -rf $script_dir/build && mkdir $script_dir/build && cd $script_dir/build
source ./../tools/scripts/env.sh
cmake ..
cd ./tests/integration/gemm/bf16
export SYCL_PROGRAM_COMPILE_OPTIONS=" -vc-codegen -vc-disable-indvars-opt -doubleGRF -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' "
make -j && ./gemm_bf16
