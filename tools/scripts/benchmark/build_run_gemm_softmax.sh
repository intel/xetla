export SYCL_PROGRAM_COMPILE_OPTIONS=" -vc-codegen -vc-disable-indvars-opt -doubleGRF -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' "

export IGC_ShaderDumpEnable=1
export IGC_DumpToCustomDir=./xetla_dumps

unset without_softmax
unset without_reduction

#without_softmax=" -DWITHOUT_SOFTMAX "
#without_reduction=" -DWITHOUT_REDUCTION "

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
repo_path=$script_dir/../../../
rm -rf $repo_path/build && mkdir $repo_path/build && cd $repo_path/build

source ../tools/scripts/env.sh
cmake .. -DCMAKE_CXX_FLAGS=" $without_softmax $without_reduction " \
&& make gemm_softmax \
&& ./examples/06_gemm_softmax/gemm_softmax
