#pragma once
#include "xetla.hpp"

using namespace gpu::xetla;

enum class fused_type : uint8_t {
    none = 0,
    bias = 1,
    bias_gelu = 2,
    res_add = 3
};

template <typename dtype_a, typename dtype_b, typename dtype_c,
        typename dtype_bias, typename dtype_res, typename dtype_acc,
        uint32_t wg_m, uint32_t wg_n, uint32_t sg_m, uint32_t sg_n,
        uint32_t sg_k, mem_layout layout_a, mem_layout layout_b,
        uint32_t global_kslicing, uint32_t local_kslicing, fused_type fused_op>
struct gemm_test_func {
    static const char *func_name() { return "gemm_test_func"; }

    static bool can_implement(dtype_a *A, dtype_b *B, dtype_c *C,
            uint32_t mat_m, uint32_t mat_n, uint32_t mat_k, uint32_t lda,
            uint32_t ldb, uint32_t ldc, dtype_bias *bias_ptr,
            dtype_res *res_ptr0, dtype_res *res_ptr1, dtype_acc *acc_ptr,
            uint32_t *cnt_ptr) {

        return true;
    }
    static inline void run(sycl::nd_item<3> &item, dtype_a *A, dtype_b *B,
            dtype_c *C, uint32_t mat_m, uint32_t mat_n, uint32_t mat_k,
            uint32_t lda, uint32_t ldb, uint32_t ldc, dtype_bias *bias_ptr,
            dtype_res *res_ptr0, dtype_res *res_ptr1, dtype_acc *acc_ptr,
            uint32_t *cnt_ptr) {}
};