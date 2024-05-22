script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
rm -rf $script_dir/build && mkdir $script_dir/build && cd $script_dir/build
source ./../tools/scripts/env.sh
cmake ..
cd ./tests/integration/gemm/bf16
make -j && ./gemm_bf16
