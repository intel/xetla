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

#ifndef GPU_XETPP_CORE_MEMORY_HPP
#define GPU_XETPP_CORE_MEMORY_HPP

#include "xetpp_core_common.hpp"

namespace gpu {
namespace xetpp {
namespace core {

namespace detail {

/// @brief lookup table for cache hint
///
///
#if XETPP_ESIMD_ENABLED
constexpr __ESIMD_ENS::cache_hint get_cache_hint(
        __XETPP_CORE_NS::cache_hint ch) {
    switch (ch) {
        case __XETPP_CORE_NS::cache_hint::none:
            return __ESIMD_ENS::cache_hint::none;
        case __XETPP_CORE_NS::cache_hint::uncached:
            return __ESIMD_ENS::cache_hint::uncached;
        case __XETPP_CORE_NS::cache_hint::cached:
            return __ESIMD_ENS::cache_hint::cached;
        case __XETPP_CORE_NS::cache_hint::write_back:
            return __ESIMD_ENS::cache_hint::write_back;
        case __XETPP_CORE_NS::cache_hint::write_through:
            return __ESIMD_ENS::cache_hint::write_through;
        case __XETPP_CORE_NS::cache_hint::streaming:
            return __ESIMD_ENS::cache_hint::streaming;
        case __XETPP_CORE_NS::cache_hint::read_invalidate:
            return __ESIMD_ENS::cache_hint::read_invalidate;
        default: return __ESIMD_ENS::cache_hint::none;
    }
}
#else
constexpr CacheHint get_cache_hint(__XETPP_CORE_NS::cache_hint ch) {
    switch (ch) {
        case __XETPP_CORE_NS::cache_hint::none: return CacheHint::Default;
        case __XETPP_CORE_NS::cache_hint::uncached: return CacheHint::Uncached;
        case __XETPP_CORE_NS::cache_hint::cached: return CacheHint::Cached;
        case __XETPP_CORE_NS::cache_hint::write_back:
            return CacheHint::WriteBack;
        case __XETPP_CORE_NS::cache_hint::write_through:
            return CacheHint::WriteThrough;
        case __XETPP_CORE_NS::cache_hint::streaming:
            return CacheHint::Streaming;
        case __XETPP_CORE_NS::cache_hint::read_invalidate:
            return CacheHint::ReadInvalidate;
        default: return CacheHint::Default;
    }
}
#endif

/// @brief lookup table for data size
///
///
#if XETPP_ESIMD_ENABLED
constexpr __ESIMD_ENS::lsc_data_size get_data_size(
        __XETPP_CORE_NS::lsc_data_size ds) {
    switch (ds) {
        case __XETPP_CORE_NS::lsc_data_size::default_size:
            return __ESIMD_ENS::lsc_data_size::default_size;
        case __XETPP_CORE_NS::lsc_data_size::u8:
            return __ESIMD_ENS::lsc_data_size::u8;
        case __XETPP_CORE_NS::lsc_data_size::u16:
            return __ESIMD_ENS::lsc_data_size::u16;
        case __XETPP_CORE_NS::lsc_data_size::u32:
            return __ESIMD_ENS::lsc_data_size::u32;
        case __XETPP_CORE_NS::lsc_data_size::u64:
            return __ESIMD_ENS::lsc_data_size::u64;
        case __XETPP_CORE_NS::lsc_data_size::u8u32:
            return __ESIMD_ENS::lsc_data_size::u8u32;
        case __XETPP_CORE_NS::lsc_data_size::u16u32:
            return __ESIMD_ENS::lsc_data_size::u16u32;
        case __XETPP_CORE_NS::lsc_data_size::u16u32h:
            return __ESIMD_ENS::lsc_data_size::u16u32h;
        default: return __ESIMD_ENS::lsc_data_size::default_size;
    }
}
#else
constexpr DataSize get_data_size(__XETPP_CORE_NS::lsc_data_size ds) {
    switch (ds) {
        case __XETPP_CORE_NS::lsc_data_size::default_size:
            return DataSize::Default;
        case __XETPP_CORE_NS::lsc_data_size::u8: return DataSize::U8;
        case __XETPP_CORE_NS::lsc_data_size::u16: return DataSize::U16;
        case __XETPP_CORE_NS::lsc_data_size::u32: return DataSize::U32;
        case __XETPP_CORE_NS::lsc_data_size::u64: return DataSize::U64;
        case __XETPP_CORE_NS::lsc_data_size::u8u32: return DataSize::U8U32;
        case __XETPP_CORE_NS::lsc_data_size::u16u32: return DataSize::U16U32;
        case __XETPP_CORE_NS::lsc_data_size::u16u32h: return DataSize::U16U32H;
        default: return DataSize::Default;
    }
}
#endif

} // namespace detail

/// @addtogroup xetpp_core_memory
/// @{

/// @brief Stateless scattered prefetch.
/// Prefetches elements located at specified address.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_load.ugm
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to prefetch per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the number of channels (platform dependent).
/// @param p       [in] is the base pointer.
/// @param offsets [in] is the zero-based offsets in bytes.
/// @param pred    [in] is predicates.
///
template <typename T, uint8_t NElts = 1,
        lsc_data_size DS = lsc_data_size::default_size,
        cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
        int N>
__XETPP_API void xetpp_prefetch_ugm(
        const T *p, xetpp_vector<uint32_t, N> offsets, xetpp_mask<N> pred = 1) {
#if XETPP_ESIMD_ENABLED
    __ESIMD_ENS::lsc_prefetch<T, NElts, detail::get_data_size(DS),
            detail::get_cache_hint(L1H), detail::get_cache_hint(L3H), N>(
            p, offsets, pred);
#else
    cm_ptr_prefetch<details::lsc_vector_size<NElts>(),
            detail::get_data_size(DS), detail::get_cache_hint(L1H),
            detail::get_cache_hint(L3H), N>(p, offsets, pred);
#endif
}

/// @brief Stateless block prefetch (transposed gather with 1 channel).
/// Prefetches elements located at specified address.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_load.ugm
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param p      [in] is the base pointer.
/// @param offset [in] is the zero-based offset in bytes.
///
template <typename T, uint8_t NElts = 1,
        lsc_data_size DS = lsc_data_size::default_size,
        cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
__XETPP_API void xetpp_prefetch_ugm(const T *p, uint64_t offset) {
#if XETPP_ESIMD_ENABLED
    __ESIMD_ENS::lsc_prefetch<T, NElts, detail::get_data_size(DS),
            detail::get_cache_hint(L1H), detail::get_cache_hint(L3H)>(
            p + (offset / sizeof(T)));
#else
    cm_ptr_prefetch<NElts, detail::get_data_size(DS),
            detail::get_cache_hint(L1H), detail::get_cache_hint(L3H)>(
            p, offset);
#endif
}

/// @brief Stateless scattered load.
/// Collects elements located at specified address and returns them
/// to a single \ref xetpp_vector object.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_load.ugm
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the number of channels (platform dependent).
/// @param p       [in] is the base pointer.
/// @param offsets [in] is the zero-based offsets in bytes.
/// @param pred    [in] is predicates.
/// @return  is a xetpp_vector of type T and size N * NElts
///
template <typename T, uint8_t NElts = 1,
        lsc_data_size DS = lsc_data_size::default_size,
        cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
        int N>
__XETPP_API xetpp_vector<T, N * NElts> xetpp_load_ugm(
        const T *p, xetpp_vector<uint32_t, N> offsets, xetpp_mask<N> pred = 1) {
#if XETPP_ESIMD_ENABLED
    return __ESIMD_ENS::lsc_gather<T, NElts, detail::get_data_size(DS),
            detail::get_cache_hint(L1H), detail::get_cache_hint(L3H), N>(
            p, offsets, pred);
#else
    return cm_ptr_load<T, details::lsc_vector_size<NElts>(),
            detail::get_data_size(DS), detail::get_cache_hint(L1H),
            detail::get_cache_hint(L3H), N>(p, offsets, pred);
#endif
}

/// @brief Stateless block load (transposed gather with 1 channel).
/// Collects elements located at specified address and returns them
/// to a single \ref xetpp_vector object.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_load.ugm
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param p      [in] is the base pointer.
/// @param offset [in] is the zero-based offset in bytes.
/// @return is a xetpp_vector of type T and size NElts
///
template <typename T, uint8_t NElts = 1,
        lsc_data_size DS = lsc_data_size::default_size,
        cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
__XETPP_API xetpp_vector<T, NElts> xetpp_load_ugm(const T *p, uint64_t offset) {
#if XETPP_ESIMD_ENABLED
    return __ESIMD_ENS::lsc_block_load<T, NElts, detail::get_data_size(DS),
            detail::get_cache_hint(L1H), detail::get_cache_hint(L3H)>(
            p + (offset / sizeof(T)));
#else
    return cm_ptr_load<T, NElts, detail::get_data_size(DS),
            detail::get_cache_hint(L1H), detail::get_cache_hint(L3H)>(
            p, offset);
#endif
}

/// @brief Stateless scattered store.
/// Writes elements to specific address.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_store.ugm
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to store per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the number of channels (platform dependent).
/// @param p       [in] is the base pointer.
/// @param offsets [in] is the zero-based offsets in bytes.
/// @param vals    [in] is values to store.
/// @param pred    [in] is predicates.
///
template <typename T, uint8_t NElts = 1,
        lsc_data_size DS = lsc_data_size::default_size,
        cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
        int N>
__XETPP_API void xetpp_store_ugm(T *p, xetpp_vector<uint32_t, N> offsets,
        xetpp_vector<T, N * NElts> vals, xetpp_mask<N> pred = 1) {
#if XETPP_ESIMD_ENABLED
    __ESIMD_ENS::lsc_scatter<T, NElts, detail::get_data_size(DS),
            detail::get_cache_hint(L1H), detail::get_cache_hint(L3H), N>(
            p, offsets, vals, pred);
#else
    cm_ptr_store<T, details::lsc_vector_size<NElts>(),
            detail::get_data_size(DS), detail::get_cache_hint(L1H),
            detail::get_cache_hint(L3H), N>(p, offsets, vals, pred);
#endif
}

/// @brief Stateless block store (transposed scatter with 1 channel).
/// Writes elements to specific address.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_store.ugm
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to store per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param p      [in] is the base pointer.
/// @param offset [in] is the zero-based offset in bytes.
/// @param vals   [in] is values to store.
///
template <typename T, uint8_t NElts = 1,
        lsc_data_size DS = lsc_data_size::default_size,
        cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
__XETPP_API void xetpp_store_ugm(
        T *p, uint64_t offset, xetpp_vector<T, NElts> vals) {
#if XETPP_ESIMD_ENABLED
    __ESIMD_ENS::lsc_block_store<T, NElts, detail::get_data_size(DS),
            detail::get_cache_hint(L1H), detail::get_cache_hint(L3H)>(
            p + (offset / sizeof(T)), vals);
#else
    cm_ptr_store<T, NElts, detail::get_data_size(DS),
            detail::get_cache_hint(L1H), detail::get_cache_hint(L3H)>(
            p, offset, vals);
#endif
}

/// @} xetpp_core_memory

} // namespace core
} // namespace xetpp
} // namespace gpu

#endif
