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

#ifndef GPU_XETPP_CORE_BASE_TYPES_HPP
#define GPU_XETPP_CORE_BASE_TYPES_HPP

#include "xetpp_core_common.hpp"

namespace gpu {
namespace xetpp {
namespace core {

/// @addtogroup xetpp_core_base_types
/// @{

/// @brief wrapper for xetpp_vector.
#if XETPP_ESIMD_ENABLED
/// Alias to ESIMD `__ESIMD_NS::simd`;
/// @tparam Ty data type in xetpp_vector
/// @tparam N  data length in xetpp_vector
///
template <typename Ty, uint32_t N>
using xetpp_vector = __ESIMD_NS::simd<Ty, N>;
#else
/// Alias to CM `vector`;
/// @tparam Ty data type in xetpp_vector
/// @tparam N  data length in xetpp_vector
///
template <typename Ty, uint32_t N>
using xetpp_vector = vector<Ty, N>;
#endif

/// @brief wrapper for xetpp_mask.
#if XETPP_ESIMD_ENABLED
/// Alias to ESIMD `__ESIMD_NS::simd_mask`;
/// @tparam N  data length in xetpp_mask
///
template <uint32_t N>
using xetpp_mask = __ESIMD_NS::simd_mask<N>;
#else
/// Alias to CM `vector<uint16_t, N>`;
/// @tparam N  data length in xetpp_mask
///
template <uint32_t N>
using xetpp_mask = vector<uint16_t, N>;
#endif

/// @brief Workaround for ESIMD reference usage.
#if XETPP_ESIMD_ENABLED
/// Alias to `auto` if go with ESIMD path;
/// @see gpu::xetpp::core::xetpp_matrix_ref gpu::xetpp::core::xetpp_vector_ref
#define _REF_ auto
#else
/// Alias to `(empty)` if go with CM path;
/// @see gpu::xetpp::core::xetpp_matrix_ref gpu::xetpp::core::xetpp_vector_ref
#define _REF_
#endif

#if XETPP_ESIMD_ENABLED
/// @brief Workaround for ESIMD vector(1D) ref type.
/// Use C++20 [concept](https://en.cppreference.com/w/cpp/language/constraints) to constrains the scope of auto.
/// @note Need to be used together with `_REF_`, i.e. `"xetpp_vector_ref _REF_"` is the full declaration of xetpp vector reference
/// @tparam Ta first tparam is reserved for auto
/// @tparam Ty data type in xetpp_vector_ref
/// @tparam N  data length in xetpp_vector_ref
///
template <typename Ta, typename Ty, uint32_t N>
concept xetpp_vector_ref
        = __ESIMD_NS::detail::is_simd_view_type_v<Ta> &&std::is_same_v<
                  typename Ta::element_type,
                  Ty> && (N == __ESIMD_NS::shape_type<Ta>::type::Size_x * __ESIMD_NS::shape_type<Ta>::type::Size_y);
#else
/// @brief wrapper for xetpp_vector_ref.
/// Alias to `vector_ref` if go with CM path;
/// @note Need to be used together with `_REF_`, i.e. `"xetpp_vector_ref _REF_"` is the full declaration of xetpp vector reference
/// @tparam Ty data type in xetpp_vector
/// @tparam N  data length in xetpp_vector
///
template <typename Ty, uint32_t N>
using xetpp_vector_ref = vector_ref<Ty, N>;
#endif

#if XETPP_ESIMD_ENABLED
/// @brief Workaround for ESIMD matrix(2D) ref type.
/// Use C++20 [concept](https://en.cppreference.com/w/cpp/language/constraints) to constrains the scope of auto.
/// @note Need to be used together with `_REF_`, i.e. `"xetpp_matrix_ref _REF_"` is the full declaration of xetpp matrix reference.
/// @tparam Ta first tparam is reserved for auto
/// @tparam Ty data type in xetpp_matrix_ref
/// @tparam N1 row num in xetpp_matrix_ref
/// @tparam N2 col num in xetpp_matrix_ref
///
template <typename Ta, typename Ty, uint32_t N1, uint32_t N2>
concept xetpp_matrix_ref
        = __ESIMD_NS::detail::is_simd_view_type_v<Ta> &&std::is_same_v<
                  typename Ta::element_type,
                  Ty> && (N1 == __ESIMD_NS::shape_type<Ta>::type::Size_y)
        && (N2 == __ESIMD_NS::shape_type<Ta>::type::Size_x);
#else
/// @brief wrapper for xetpp_matrix_ref.
/// Alias to `matrix_ref` if go with CM path;
/// @note Need to be used together with `_REF_`, i.e. `"xetpp_matrix_ref _REF_"` is the full declaration of xetpp matrix reference.
/// @tparam Ty data type in xetpp_matrix_ref
/// @tparam N1 row num in xetpp_matrix_ref
/// @tparam N2 col num in xetpp_matrix_ref
///
template <typename Ty, uint32_t N1, uint32_t N2>
using xetpp_matrix_ref = matrix_ref<Ty, N1, N2>;
#endif

/// @} xetpp_core_base_types

} // namespace core
} // namespace xetpp
} // namespace gpu

#endif
