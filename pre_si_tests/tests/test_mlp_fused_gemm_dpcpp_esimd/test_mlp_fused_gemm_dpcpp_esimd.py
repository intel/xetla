import pytest
from src.marker import setup_cmake

from src.pytest_basic import PyTestBase
from src.utils.path_and_file import in_sub_dir


class TestMlpFusedGemmDpcppEsimd(PyTestBase):
    @setup_cmake(target="mlp_fused_gemm")
    def test_mlp_fused_fp16gemm_dpcpp_esimd(self, fulsim, cmake, wl):
        dir = "./examples/mlp_fused_gemm_performance" if not self.is_dry_run else "."
        with in_sub_dir(dir):
            cmd = f"./{cmake.build_target_name()} {wl._fp16_cmd}"
            assert self.run_cmd(cmd, timeout_in_minutes=120) == 0
