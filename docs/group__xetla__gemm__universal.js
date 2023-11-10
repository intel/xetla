var group__xetla__gemm__universal =
[
    [ "gpu::xetla::kernel::gemm_universal_t< dispatch_policy, gemm_t, epilogue_t, enable >", "classgpu_1_1xetla_1_1kernel_1_1gemm__universal__t.html", null ],
    [ "gpu::xetla::kernel::group_swizzle_default< arch_tag_ >", "structgpu_1_1xetla_1_1kernel_1_1group__swizzle__default.html", [
      [ "group_swizzle_default", "structgpu_1_1xetla_1_1kernel_1_1group__swizzle__default.html#aa9bcb5743266b53d32f88a4138e02905", null ]
    ] ],
    [ "gpu::xetla::kernel::group_swizzle_snake< wg_num_n_, arch_tag_ >", "structgpu_1_1xetla_1_1kernel_1_1group__swizzle__snake.html", [
      [ "group_swizzle_snake", "structgpu_1_1xetla_1_1kernel_1_1group__swizzle__snake.html#a8fb2b1a25f933de3a0415f22eba5a92e", null ]
    ] ],
    [ "gpu::xetla::kernel::dispatch_policy_default< group_swizzle_policy_ >", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__default.html", [
      [ "group_swizzle_policy", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__default.html#a60a835b549bbf55b4f5da5f859121d08", null ]
    ] ],
    [ "gpu::xetla::kernel::dispatch_policy_kslicing< group_swizzle_policy_, global_ratio_, local_ratio_ >", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__kslicing.html", [
      [ "group_swizzle_policy", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__kslicing.html#a6870db0231bd16d296ed9c61146c1ff8", null ]
    ] ],
    [ "gpu::xetla::kernel::dispatch_policy_stream_k< arch_tag_ >", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html", [
      [ "dispatch_policy_stream_k", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a9b62230c164f4282338683f1e7b9fc28", null ],
      [ "dispatch_policy_stream_k", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a84502dbf8701f2ae1bb228a49fe6c3d4", null ],
      [ "get_first_group_idx", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a5b8920c247dcd7fab5447c84cfd7e6c1", null ],
      [ "get_group_range", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a9f4e54822b662cf5bdbccbd562ecd354", null ],
      [ "get_groups", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#adb47563d3c6b646163b176342c6f4ddf", null ],
      [ "get_iter_extents", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a0ef915c8beb58aac3736fb6b1a480352", null ],
      [ "get_iters_per_tile", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a7ff0392c6263bfb0f024b1e81b4f20ed", null ],
      [ "get_num_active_groups", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#abb847335dda6d59bc9738ba880f6bd0e", null ],
      [ "get_sk_groups_per_region", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#ae4f2c2f156b2a8c44f88eef340d327e4", null ],
      [ "get_sk_iters_per_normal_group", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a4a8ea08be3513b85adc50f973e9d7034", null ],
      [ "get_sk_regions", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a9563d198d63ca4058109495796bfcf16", null ],
      [ "get_sk_tile_idx", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a0993b279d313beb59fedc0694c6556f7", null ],
      [ "get_sk_workgroups", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a04c2f5f851ab8c3f13c9e0c94266b653", null ],
      [ "get_tile_offsets", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a41ed643acb3a1c0324e8eac534ab5541", null ],
      [ "avail_xecores", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#afae0e7fe27987d6956890119366858b9", null ],
      [ "div_mod_iters_per_tile", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a10bcc1dd62806a9aa12e0ade98206c2d", null ],
      [ "div_mod_sk_groups_per_region", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a61c1641d95e9f36396cbd5babec34ed6", null ],
      [ "div_mod_sk_iters_per_big_group", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a50f613b3d6e9914ce743bd787973e8d4", null ],
      [ "div_mod_sk_iters_per_normal_group", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a9f53a881e36aa2b34d878c3648fcdc03", null ],
      [ "div_mod_sk_iters_per_region", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a02fe8f650e6aafae603bc24ce2440c59", null ],
      [ "div_mod_sk_regions", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a6d8f13f212cff45c6957ccfa571745e5", null ],
      [ "div_mod_tiles_m", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a04799510b60ce10be3a8c5573ddedb40", null ],
      [ "div_mod_tiles_n", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a5d5403e6fbdb4c8a578bd6cad9d1cc9d", null ],
      [ "dp_groups", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a8628b6625009709bd4c9ebe7df4232fe", null ],
      [ "matrix_k", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#ac91f9b5bc6fe3c7d9a6d1169515c274b", null ],
      [ "matrix_m", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a78f0e5b3bd236f62bcd97a28ca349639", null ],
      [ "matrix_n", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a422504d8bd3b71f08cb4bbbb33f263c9", null ],
      [ "num_workgroups", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a17c5ec491d2634d75baf6e6b25de449e", null ],
      [ "sg_tile_m", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#abd989ba9a2c555c1eea3763037e46b1d", null ],
      [ "sg_tile_n", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a4d88d3de1fcb2fe5e31c3370ffd278a7", null ],
      [ "sk_big_groups_per_region", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a33bba9efe22a77d2aeb3688293fd9182", null ],
      [ "sk_groups_per_region", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#aef0cf57885988abbee0a49a6f96e4929", null ],
      [ "sk_iters_per_region", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a6de0d048425853a8fe102e50cd70ff09", null ],
      [ "sk_regions", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#aa5f296d2658eee36d8ae5d82bf18bf3c", null ],
      [ "sk_tiles", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#aa7d86bcbe40176339bee44dc56c9a460", null ],
      [ "sk_waves", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#aa62bae525e3ec71e877cb754a3a94fba", null ],
      [ "wg_tile_k", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a780d0506d6910a39e4a5350d71f67ff2", null ],
      [ "wg_tile_m", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#adbea2a36797678a84c6aa2591482e59a", null ],
      [ "wg_tile_n", "structgpu_1_1xetla_1_1kernel_1_1dispatch__policy__stream__k.html#a6c87259f8eeebca2fbef8ba9116ba634", null ]
    ] ],
    [ "gpu::xetla::kernel::gemm_universal_t< dispatch_policy_default< group_swizzle_ >, gemm_t_, epilogue_t_, std::enable_if_t<(group_swizzle_::arch_tag==gpu_arch::Xe)> >", "classgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__default_3_01group__swizz08bd19f9866e2b3c01dc964b05f0cd85.html", [
      [ "arguments_t", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__default_3_01group__swizfba6d51ee66d93e1d7cdebfe7c9ba60e.html", [
        [ "arguments_t", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__default_3_01group__swizfba6d51ee66d93e1d7cdebfe7c9ba60e.html#aef1c76a66fee0ce988423daea2069daa", null ],
        [ "arguments_t", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__default_3_01group__swizfba6d51ee66d93e1d7cdebfe7c9ba60e.html#a47a3c62b21927b80bce1af1d9d8607bc", null ],
        [ "arguments_t", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__default_3_01group__swizfba6d51ee66d93e1d7cdebfe7c9ba60e.html#a7b8690a748689bdcf759b449cff4edeb", null ],
        [ "operator=", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__default_3_01group__swizfba6d51ee66d93e1d7cdebfe7c9ba60e.html#a6d2c5d3cacaa0aa06f44a0cf34d2d4c5", null ],
        [ "epilogue_args", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__default_3_01group__swizfba6d51ee66d93e1d7cdebfe7c9ba60e.html#ab4d6d453c7b88fc963cb850640c7d7cb", null ],
        [ "matA_base", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__default_3_01group__swizfba6d51ee66d93e1d7cdebfe7c9ba60e.html#a749bbc5edcf56f84bb31468d89b5d16c", null ],
        [ "matA_ld", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__default_3_01group__swizfba6d51ee66d93e1d7cdebfe7c9ba60e.html#a5eebf43129275be833232cbe5d74d4e9", null ],
        [ "matB_base", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__default_3_01group__swizfba6d51ee66d93e1d7cdebfe7c9ba60e.html#a7d8e00936c4aaabecbf01756286bd202", null ],
        [ "matB_ld", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__default_3_01group__swizfba6d51ee66d93e1d7cdebfe7c9ba60e.html#a699208693da4746984d21f0db7d42034", null ],
        [ "matC_base", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__default_3_01group__swizfba6d51ee66d93e1d7cdebfe7c9ba60e.html#a995e84e39759aba6dd31715efacb8548", null ],
        [ "matC_ld", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__default_3_01group__swizfba6d51ee66d93e1d7cdebfe7c9ba60e.html#ade87a375a27d4759a05efd742b4a6ae0", null ],
        [ "matrix_k", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__default_3_01group__swizfba6d51ee66d93e1d7cdebfe7c9ba60e.html#a9f77acd18e41d398fab03bb26517bd37", null ],
        [ "matrix_m", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__default_3_01group__swizfba6d51ee66d93e1d7cdebfe7c9ba60e.html#adf81ab5226702269faa36c2ce0c171a3", null ],
        [ "matrix_n", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__default_3_01group__swizfba6d51ee66d93e1d7cdebfe7c9ba60e.html#a8248275fb4a14e810b3e7576cdfe225b", null ]
      ] ],
      [ "operator()", "classgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__default_3_01group__swizz08bd19f9866e2b3c01dc964b05f0cd85.html#ad186813995f1f7e46b8af81b58d0ba8d", null ]
    ] ],
    [ "gpu::xetla::kernel::gemm_universal_t< dispatch_policy_kslicing< group_swizzle_, num_global_kslicing_, num_local_kslicing_ >, gemm_t_, epilogue_t_, std::enable_if_t<(group_swizzle_::arch_tag==gpu_arch::Xe)> >", "classgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__kslicing_3_01group__swiz2f6612c2aa0f84e417c6c4e50a6c8f9a.html", [
      [ "arguments_t", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__kslicing_3_01group__swi1c69465c60dfb436e25d7ded8490e71b.html", [
        [ "arguments_t", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__kslicing_3_01group__swi1c69465c60dfb436e25d7ded8490e71b.html#a49860f127640a63c070b89e14d7f4f5b", null ],
        [ "arguments_t", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__kslicing_3_01group__swi1c69465c60dfb436e25d7ded8490e71b.html#ad89696e97caa060bd65ecf1c2a401718", null ],
        [ "arguments_t", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__kslicing_3_01group__swi1c69465c60dfb436e25d7ded8490e71b.html#afe8bd2bb532dbf53a88d08fa2eb92c0c", null ],
        [ "operator=", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__kslicing_3_01group__swi1c69465c60dfb436e25d7ded8490e71b.html#afac49415cb2a51bda170bf216f1be551", null ],
        [ "acc_base", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__kslicing_3_01group__swi1c69465c60dfb436e25d7ded8490e71b.html#a73f9cc217acb28f5b468ab566b1006ef", null ],
        [ "cnt_base", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__kslicing_3_01group__swi1c69465c60dfb436e25d7ded8490e71b.html#a193736e8e30916d14d25a9c2809e3c58", null ],
        [ "epilogue_args", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__kslicing_3_01group__swi1c69465c60dfb436e25d7ded8490e71b.html#a263c116b24b37220e5831bbdd4ba6b5f", null ],
        [ "matA_base", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__kslicing_3_01group__swi1c69465c60dfb436e25d7ded8490e71b.html#a793d67c17909b776dff29c18647152cf", null ],
        [ "matA_ld", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__kslicing_3_01group__swi1c69465c60dfb436e25d7ded8490e71b.html#ae05b287c889c9e0a78b0bd5905a500d3", null ],
        [ "matB_base", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__kslicing_3_01group__swi1c69465c60dfb436e25d7ded8490e71b.html#ad0fff44009c5b7967b8ab36ea627d51a", null ],
        [ "matB_ld", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__kslicing_3_01group__swi1c69465c60dfb436e25d7ded8490e71b.html#a7a89a173ac07682ba3965b5f09514c9e", null ],
        [ "matC_base", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__kslicing_3_01group__swi1c69465c60dfb436e25d7ded8490e71b.html#a85949fae701fe9733273e789eb58daa6", null ],
        [ "matC_ld", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__kslicing_3_01group__swi1c69465c60dfb436e25d7ded8490e71b.html#a41bf467bed2fccbd096774191e659e3c", null ],
        [ "matrix_k", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__kslicing_3_01group__swi1c69465c60dfb436e25d7ded8490e71b.html#ab3969653e70c81482dad4f3fddf33d67", null ],
        [ "matrix_m", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__kslicing_3_01group__swi1c69465c60dfb436e25d7ded8490e71b.html#ae7acee657ac455c9655dad90b9273bcb", null ],
        [ "matrix_n", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__kslicing_3_01group__swi1c69465c60dfb436e25d7ded8490e71b.html#abf96530b8a6cfe34bb5e7e935e34f377", null ]
      ] ],
      [ "operator()", "classgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__kslicing_3_01group__swiz2f6612c2aa0f84e417c6c4e50a6c8f9a.html#a1d28e29a20c2f56d1697000778cdd708", null ]
    ] ],
    [ "gpu::xetla::kernel::gemm_universal_t< dispatch_policy_stream_k< gpu_arch::Xe >, gemm_t_, epilogue_t_ >", "classgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch_1068cc1829ada3a87a490c1d50a71487.html", [
      [ "arguments_t", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch44ec53fb5fda79c5dea48a2afb335e8b.html", [
        [ "arguments_t", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch44ec53fb5fda79c5dea48a2afb335e8b.html#a0bc93ce49bbba414e5a8cc609ee399d9", null ],
        [ "arguments_t", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch44ec53fb5fda79c5dea48a2afb335e8b.html#a19c568bc6694894efaa3e6e407a93864", null ],
        [ "arguments_t", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch44ec53fb5fda79c5dea48a2afb335e8b.html#a9076ae87823d9966c0bb3128beec1f3d", null ],
        [ "operator=", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch44ec53fb5fda79c5dea48a2afb335e8b.html#a04821f710323387a5cedd72a1297c05f", null ],
        [ "epilogue_args", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch44ec53fb5fda79c5dea48a2afb335e8b.html#adfb71bf8ce9b3acf6999ae675a9b3e4d", null ],
        [ "matA_base", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch44ec53fb5fda79c5dea48a2afb335e8b.html#ad58f2b6925ad78a85b16163d22d4d0cb", null ],
        [ "matA_ld", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch44ec53fb5fda79c5dea48a2afb335e8b.html#a0e465e9fc9ca1b0c387ce7109f0393a0", null ],
        [ "matatomic_sync_base", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch44ec53fb5fda79c5dea48a2afb335e8b.html#a8a8d76fcc5a9f4bdd0c8b6893e34ec7d", null ],
        [ "matatomic_sync_ld", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch44ec53fb5fda79c5dea48a2afb335e8b.html#a64885f0ccc5e269d36e98b129d83f05f", null ],
        [ "matB_base", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch44ec53fb5fda79c5dea48a2afb335e8b.html#a37d7955ec4cb5ad0dcee39d60c3cd344", null ],
        [ "matB_ld", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch44ec53fb5fda79c5dea48a2afb335e8b.html#a4ca4089bbb60c3a0e1e8804fefc471d8", null ],
        [ "matC_base", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch44ec53fb5fda79c5dea48a2afb335e8b.html#a0b08087c9b6a303272968edbaa496dde", null ],
        [ "matC_ld", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch44ec53fb5fda79c5dea48a2afb335e8b.html#a3bc8427212a04a803f83b733c962a424", null ],
        [ "matD_base", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch44ec53fb5fda79c5dea48a2afb335e8b.html#a12e19bc5a9226f96b702ea348a1a4a6b", null ],
        [ "matD_ld", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch44ec53fb5fda79c5dea48a2afb335e8b.html#a590e9d966fc3589612c2f3a92938dafe", null ],
        [ "matrix_k", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch44ec53fb5fda79c5dea48a2afb335e8b.html#aa65bc76f8f259ce03e69800ce84d6ba3", null ],
        [ "matrix_m", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch44ec53fb5fda79c5dea48a2afb335e8b.html#a357648b09ae750cc6964c9d4132710c1", null ],
        [ "matrix_n", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch44ec53fb5fda79c5dea48a2afb335e8b.html#a6f665fc91f5a82973876f3690c996130", null ],
        [ "stream_k_args", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch44ec53fb5fda79c5dea48a2afb335e8b.html#a7d87f1d124f5d28f5abf14694f3cb90d", null ]
      ] ],
      [ "TileWorkDesc", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch421c1a117893f171c99088210ef8b82b.html", [
        [ "iter_begin", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch421c1a117893f171c99088210ef8b82b.html#a453b87eba719acdd2ff561783cb37131", null ],
        [ "k_begin", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch421c1a117893f171c99088210ef8b82b.html#a1fee5e28e42acf7e222f891f7b057a59", null ],
        [ "k_end", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch421c1a117893f171c99088210ef8b82b.html#a9cc33867f9159578e959aad793c734b1", null ],
        [ "k_iters_remaining", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch421c1a117893f171c99088210ef8b82b.html#a1ecc9db96f35718729ebec22ab961444", null ],
        [ "tile_offset_m", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch421c1a117893f171c99088210ef8b82b.html#afe0d12d1328b5606a01770844e0af952", null ],
        [ "tile_offset_n", "structgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch421c1a117893f171c99088210ef8b82b.html#a5ebbc892f58230fbed87ccce05d57fe3", null ]
      ] ],
      [ "operator()", "classgpu_1_1xetla_1_1kernel_1_1gemm__universal__t_3_01dispatch__policy__stream__k_3_01gpu__arch_1068cc1829ada3a87a490c1d50a71487.html#abb34ed768a1b01135d29116c00902674", null ]
    ] ]
];