unset SYCL_PROGRAM_COMPILE_OPTIONS
unset sycl_compiler_path
unset gpu_driver_path
unset without_softmax
unset without_reduction

export ZE_AFFINITY_MASK=0

export SYCL_PROGRAM_COMPILE_OPTIONS=" -vc-codegen -vc-disable-indvars-opt -doubleGRF -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' "


sycl_compiler_path=/opt/intel_validation/oneapi/compiler/2024.0/

# https://ubit-gfx.intel.com/build/19168301/artifacts
#gpu_driver_path=/opt/cutlass/gpu_driver/agama-996.6/extract/
#gpu_driver_path=/opt/cutlass/gpu_driver/gfx-driver-ci-comp_igc-27004/extract/

export CPATH=$sycl_compiler_path/include:$sycl_compiler_path/include/sycl:/opt/intel_validation/oneapi/2024.0/include/
export CC=${sycl_compiler_path}/bin/icx
export CXX=${sycl_compiler_path}/bin/icpx

export LIBRARY_PATH=$gpu_driver_path/usr/lib/x86_64-linux-gnu/:$sycl_compiler_path/lib/:/opt/intel_validation/oneapi/2024.0/lib/
export LD_LIBRARY_PATH=$LIBRARY_PATH
export MKLROOT=/opt/intel_validation/oneapi/mkl/2024.0/

export IGC_ShaderDumpEnable=1
export IGC_DumpToCustomDir=./xetla_dumps

#without_softmax=" -DWITHOUT_SOFTMAX "
#without_reduction=" -DWITHOUT_REDUCTION "

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
repo_path=$script_dir/../../../
rm -rf $repo_path/build && mkdir $repo_path/build && cd $repo_path/build

source ../tools/scripts/env.sh
cmake .. -DCMAKE_CXX_FLAGS=" $without_softmax $without_reduction " \
&& make gemm_softmax \
&& ./examples/06_gemm_softmax/gemm_softmax
