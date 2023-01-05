#!/usr/bin/env bash
script_path="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
test_framework_path="$script_path/drivers_gpu_compute_workloads"

PYTHONPATH="$test_framework_path" pytest "$@"
