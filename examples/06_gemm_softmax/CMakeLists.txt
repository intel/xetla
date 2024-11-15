
set(TARGET gemm_softmax)

set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} -fsycl)
set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} -fsycl-targets=spir64_gen)

# disable loop invariance optimization, this is for performance
set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -vc-disable-indvars-opt")
# For large registers mode, enable 256 registers for kernels
set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -doubleGRF")
# For registers usage verbose at AOT
set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -Xfinalizer -printregusage")
# Enable bank conflict reduction.
set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -Xfinalizer -enableBCR")
# Optimization to reduce the tokens used for DPAS instruction.
set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -Xfinalizer -DPASTokenReduction")

set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} -Xs)
set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} "-device pvc -options '${XETLA_OFFLINE_OPTIONS} ' ")

add_executable(${TARGET} gemm_softmax.cpp)

# AOT mode, uncommnets the following 2 lines. JIT mode, commnets the following 2 lines
target_link_options(${TARGET} PRIVATE ${XETLA_KERNEL_FLAGS})
target_compile_options(${TARGET} PRIVATE ${XETLA_KERNEL_FLAGS})

