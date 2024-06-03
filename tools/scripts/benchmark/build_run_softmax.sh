script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
repo_path=$script_dir/../../../
rm -rf $repo_path/build && mkdir $repo_path/build && cd $repo_path/build
source ../tools/scripts/env.sh
cmake .. && make softmax && ./tests/integration/softmax/softmax
