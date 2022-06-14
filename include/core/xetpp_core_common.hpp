/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @file
/// C++ API

#ifndef GPU_XETPP_CORE_COMMON_HPP
#define GPU_XETPP_CORE_COMMON_HPP

/// @addtogroup xetpp_core
/// @{

/// @brief KERNEL_MAIN macro.
#if XETPP_ESIMD_ENABLED
/// Alias to ESIMD `"SYCL_ESIMD_KERNEL"`;
///
#define KERNEL_MAIN SYCL_ESIMD_KERNEL
#else
/// Alias to CM `"_GENX_MAIN_"`;
///
#define KERNEL_MAIN _GENX_MAIN_
#endif

/// @brief KERNEL_FUNC macro.
#if XETPP_ESIMD_ENABLED
/// Alias to ESIMD `"SYCL_ESIMD_FUNCTION SYCL_EXTERNAL"`;
///
#define KERNEL_FUNC SYCL_ESIMD_FUNCTION SYCL_EXTERNAL
#else
/// Alias to CM `"_GENX_"`;
///
#define KERNEL_FUNC _GENX_
#endif

/// @} xetpp_core

#define __XETPP_CORE_NS gpu::xetpp::core
#define __XETPP_API inline

#if XETPP_ESIMD_ENABLED
#define __ESIMD_EXT_NS sycl::ext::intel::experimental::esimd

#ifndef __ESIMD_NS
#define __ESIMD_NS sycl::ext::intel::esimd
#endif
#endif

#endif
