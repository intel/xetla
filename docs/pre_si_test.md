# Pre-Silicon Test

The Xetla CI runs two pre-silicon tests.

1. The Pre PR CI runs through the native rasty
2. The Post PR CI runs through the cobalt.

## Pre PR CI

The manual steps to duplicate the Pre PR CI in the container is

`bash ci_docker/unit_test.sh` 

## Post PR CI

The post-PR CI uses the test framework of [drivers.gpu.compute.workloads](https://github.com/intel-sandbox/drivers.gpu.compute.workloads) repo. 

To manually duplicate, you need first to clone that repo locally.

```text
cd pre_si_tests
bash set_up_pre_si_test.sh
```

To run the tests, you can use `cobalt_pytest.sh,` which is just a wrapper of the pytest. And all options are the same as the `drivers.gpu.compute.workloads` repo.

For example:

This command runs all test cases under the folder tests/test_xetla using the cobalt.

`cobalt_pytest.sh tests/test_xetla --fulsim=cobalt --hw=pvcxt.b0.512.1t --wl=ci`

This command uses the `-k` option to run the cm_esimd_vadd_test only.
cobalt_pytest.sh tests/test_xetla_xetla --fulsim=cobalt --hw=pvcxt.b0.512.1t --wl=ci -k cm_esimd_vadd_test`

