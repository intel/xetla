import json
from pathlib import Path
import pytest

from src.pytest_basic import PyTestBase


def get_all_unit_test_caes():
    # build the project and use ctest --show-only=json-v1 > ctest_cases.json to update it.
    # if just want to run single test, use -k option with pytest, like -k fp16_gemm
    with open(Path(__file__).parent / "ctest_cases.json") as f:
        ctest_info = json.load(f)
    return [test["name"] for test in ctest_info["tests"]]


class TestXetlaUnitTest(PyTestBase):
    @pytest.mark.parametrize(
        argnames="cmake_target", argvalues=get_all_unit_test_caes()
    )
    def test_xetla_unit_test(self, cmake_target, fulsim, cmake, wl):
        cmd = f"ctest --tests-regex ^{cmake_target}$ --verbose"
        assert self.run_cmd(cmd, timeout_in_minutes=30) == 0
