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

#ifndef GPU_XETPP_CORE_BASE_OPS_HPP
#define GPU_XETPP_CORE_BASE_OPS_HPP

namespace gpu {
namespace xetpp {
namespace core {

/// @addtogroup xetpp_core_base_ops
/// @{

/// @brief xetpp format.
#if XETPP_ESIMD_ENABLED
/// Alias to ESIMD `.template bit_cast_view<...>()`;
/// @note usage:
/// ```
/// [xetpp_vector|xetpp_vector_ref|xetpp_matrix_ref].xetpp_format<type>(): returns a reference to the calling object interpreted as a new xetpp_vector_ref (1D)
///
/// [xetpp_vector|xetpp_vector_ref|xetpp_matrix_ref].xetpp_format<type, rows, columns>(): returns a reference to the calling object interpreted as a new xetpp_matrix_ref (2D)
/// ```
///
#define xetpp_format template bit_cast_view
#else
/// Alias to CM `.format<...>()`;
/// @note usage:
/// ```
/// [xetpp_vector|xetpp_vector_ref|xetpp_matrix_ref].xetpp_format<type>(): returns a reference to the calling object interpreted as a new xetpp_vector_ref (1D)
///
/// [xetpp_vector|xetpp_vector_ref|xetpp_matrix_ref].xetpp_format<type, rows, columns>(): returns a reference to the calling object interpreted as a new xetpp_matrix_ref (2D)
/// ```
///
#define xetpp_format format
#endif

/// @brief xetpp select.
#if XETPP_ESIMD_ENABLED
/// Alias to ESIMD `.template select<...>()`;
/// @note usage:
/// ```
/// [xetpp_vector|xetpp_vector_ref].xetpp_select<size, stride>(uint16_t offset=0): returns a reference to the sub-vector starting from the offset-th element
///
/// [xetpp_matrix_ref].xetpp_select<size_y, stride_y, size_x, stride_x>(uint16_t offset_y=0, uint16_t offset_x=0): returns a reference to the sub-matrix starting from the (offset_y, offset_x)-th element
/// ```
///
#define xetpp_select template select
#else
/// Alias to CM `.select<...>()`;
/// @note usage:
/// ```
/// [xetpp_vector|xetpp_vector_ref].xetpp_select<size, stride>(uint16_t offset=0): returns a reference to the sub-vector starting from the offset-th element
///
/// [xetpp_matrix_ref].xetpp_select<size_y, stride_y, size_x, stride_x>(uint16_t offset_y=0, uint16_t offset_x=0): returns a reference to the sub-matrix starting from the (offset_y, offset_x)-th element
/// ```
///
#define xetpp_select select
#endif

/// @brief xetpp merge.
/// Alias to `.merge(...)`. Replaces part of the underlying data with the one taken from the other object according to a mask.
/// @note usage:
/// ```
/// [xetpp_vector|xetpp_vector_ref].xetpp_merge(xetpp_vector<Ty, N>Val, xetpp_mask<N>mask): only elements in lanes with non-zero mask predicate are assigned from corresponding Val elements
///
/// [xetpp_vector|xetpp_vector_ref].xetpp_merge(xetpp_vector<Ty, N>Val1, xetpp_vector<Ty, N>Val2, xetpp_mask<N>mask): non-zero in a mask's lane tells to take corresponding element from Val1, zero - from Val2.
/// ```
///
#define xetpp_merge merge

// TODO add replicate, iselect

/// @} xetpp_core_base_ops

} // namespace core
} // namespace xetpp
} // namespace gpu

#endif
