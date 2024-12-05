script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
repo_path=$script_dir/../../../
mkdir -p $repo_path/build
rm -rf $repo_path/build/* && cd $repo_path/build

clear

unset SYCL_PROGRAM_COMPILE_OPTIONS
unset sycl_compiler_path
unset gpu_driver_path
unset without_softmax
unset without_reduction
unset disable_prefetch
unset disable_gemm

export ZE_AFFINITY_MASK=0
export SYCL_CACHE_PERSISTENT=0
export SYCL_CACHE_IN_MEM=0

#enable doubleGRF for shapes 512*64*512, 1024*64*1024, 2048*64*2048
#export SYCL_PROGRAM_COMPILE_OPTIONS=" -vc-codegen -vc-disable-indvars-opt -doubleGRF -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' "
#disable doubleGRF for shapes 2048*64*2048, 4096*64*4096, 8192*64*8192, 16384*64*16384
#export SYCL_PROGRAM_COMPILE_OPTIONS=" -vc-codegen -vc-disable-indvars-opt -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' "


sycl_compiler_path=/opt/cutlass/compiler/1008/
export MKLROOT=/opt/intel/oneapi/mkl/2024.2/

# https://ubit-gfx.intel.com/build/19168301/artifacts
#gpu_driver_path=/opt/cutlass/gpu_driver/agama-996.6/extract/
gpu_driver_path=/opt/cutlass/gpu_driver/gfx-driver-ci-comp_igc-27004/extract/

export CPATH=$sycl_compiler_path/include:$sycl_compiler_path/include/sycl:$MKLROOT/include/
export CC=$sycl_compiler_path/bin/clang
export CXX=$sycl_compiler_path/bin/clang++

export LIBRARY_PATH=$gpu_driver_path/usr/lib/x86_64-linux-gnu/:$sycl_compiler_path/lib/:$MKLROOT/lib/
export LD_LIBRARY_PATH=$LIBRARY_PATH

export IGC_ShaderDumpEnable=1
export IGC_DumpToCustomDir=$repo_path/build/xetla_dumps

#disable_gemm=" -DDISABLE_GEMM "
disable_prefetch=" -DDISABLE_GEMM_PREFETCH "
#without_softmax=" -DWITHOUT_SOFTMAX "
#without_reduction=" -DWITHOUT_REDUCTION "

source $repo_path/tools/scripts/env_debug.sh
cmake .. -DCMAKE_CXX_FLAGS=" $without_softmax $without_reduction $disable_prefetch $disable_gemm " \
&& make gemm_softmax && ./examples/06_gemm_softmax/gemm_softmax


#unitrace --chrome-kernel-logging -k -i 20 -o xetla_pvc_gemm.csv ./examples/06_gemm_softmax/gemm_softmax
#unitrace -k -i 20 --chrome-kernel-logging -o xetla.csv ./examples/06_gemm_softmax/gemm_softmax
#unitrace -k --device-timing --kernel-submission --device-timeline --chrome-kernel-logging --chrome-device-logging --chrome-no-thread-on-device --chrome-no-engine-on-device -i 20 ./examples/06_gemm_softmax/gemm_softmax -o xetla.csv

#python3 ~/workspace/cutlass/unitrace/tools/unitrace/scripts/analyzeperfmetrics.py -s $IGC_DumpToCustomDir -t "XVE Stalls by Instruction" $csv_file -o ${csv_file}.pdf

