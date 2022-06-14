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

#ifndef GPU_XETPP_CORE_HPP
#define GPU_XETPP_CORE_HPP

/// @defgroup xetpp_core XeTPP Core
/// This is a low-level API wrapper for [ESIMD](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md) and CM.

/// The macro XETPP_ESIMD_ENABLED is used to control if we go with ESIMD path or CM path.
/// Some terminologies used in the API documentation:
/// - *word* - 2 bytes.
/// - *dword* ("double word") - 4 bytes.
/// - *qword* ("quad word") - 8 bytes.
/// - *oword* ("octal word") - 16 bytes.
///

/// @addtogroup xetpp_core
/// @{

/// @defgroup xetpp_core_base_types Base types.
/// Defines vector, vector reference and matrix reference data types.

/// @defgroup xetpp_core_base_ops Base ops.
/// Defines base ops for vector, vector reference and matrix reference data types.

/// @defgroup xetpp_core_memory Memory access APIs.
/// Defines XeTPP APIs to access memory, including read, write and atomic.

/// @defgroup xetpp_core_math Math operation APIs.
/// Defines math operations on XeTPP vector data types.

/// @defgroup xetpp_core_bitmanip Bit and mask manipulation APIs.
/// Defines bitwise operations.

/// @defgroup xetpp_core_conv Explicit conversion APIs.
/// Defines explicit conversions (with and without saturation), truncation etc. between XeTPP vector types.

/// @defgroup xetpp_core_raw_send Raw send APIs.
/// Implements the \c send instruction to send messages to variaous components
/// of the Intel(R) processor graphics, as defined in the documentation at [here](https://01.org/sites/default/files/documentation/intel-gfx-prm-osrc-icllp-vol02a-commandreference-instructions_2.pdf)

/// @defgroup xetpp_core_misc Miscellaneous XeTPP convenience functions.
/// Wraps some useful functions.

/// @} xetpp_core

#include "xetpp_core_base_consts.hpp"
#include "xetpp_core_base_ops.hpp"
#include "xetpp_core_base_types.hpp"
#include "xetpp_core_bit_mask_manip.hpp"
#include "xetpp_core_common.hpp"
#include "xetpp_core_explicit_conv.hpp"
#include "xetpp_core_math_fma.hpp"
#include "xetpp_core_math_mma.hpp"
#include "xetpp_core_memory.hpp"
#include "xetpp_core_misc.hpp"
#include "xetpp_core_raw_send.hpp"

#endif
