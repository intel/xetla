/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
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

#pragma once

#include "common/utils/common.hpp"

namespace gpu::xetla {

/// @addtogroup xetla_util_tensor_load_store
/// @{

///
/// @brief Description of nd tensor descriptor for load and store.
/// Structure is defined in [here](https://gfxspecs.intel.com/Predator/Home/Index/63986).
///
using xetla_tdescriptor = xetla_vector<uint32_t, 16>;

/// @brief Alias to xetla_vector<uint32_t, 16> reference.
#define xetla_tdescriptor_ref xetla_vector_ref<uint32_t, 16> __REF__

/// @} xetla_util_tensor_load_store

namespace detail {
__XETLA_API void xetla_set_tensor_base_address(
        xetla_tdescriptor_ref desc, uint64_t base_address) {
    desc.xetla_format<uint64_t>().xetla_select<1, 1>(0) = base_address;
}
__XETLA_API void xetla_set_tensor_base_address(
        xetla_tdescriptor_ref desc, uint32_t base_address) {
    desc.xetla_format<uint32_t>().xetla_select<1, 1>(0) = base_address;
}
__XETLA_API void xetla_set_tensor_width_x(
        xetla_tdescriptor_ref desc, uint32_t width_x) {
    desc.xetla_format<uint32_t>().xetla_select<1, 1>(2) = width_x;
}

__XETLA_API void xetla_set_tensor_width_y(
        xetla_tdescriptor_ref desc, uint32_t width_y) {
    desc.xetla_format<uint32_t>().xetla_select<1, 1>(3) = width_y;
}

__XETLA_API void xetla_set_tensor_pitch_x(
        xetla_tdescriptor_ref desc, uint32_t pitch_x) {
    desc.xetla_format<uint32_t>().xetla_select<1, 1>(4) = pitch_x;
}

__XETLA_API void xetla_set_tensor_offset_x(
        xetla_tdescriptor_ref desc, int32_t offset_x) {
    desc.xetla_format<int32_t>().xetla_select<1, 1>(5) = offset_x;
}

__XETLA_API void xetla_set_tensor_offset_y(
        xetla_tdescriptor_ref desc, int32_t offset_y) {
    desc.xetla_format<int32_t>().xetla_select<1, 1>(6) = offset_y;
}

__XETLA_API int32_t xetla_get_tensor_width_x(xetla_tdescriptor desc) {
    return desc.xetla_format<int32_t>().xetla_select<1, 1>(2)[0];
}

__XETLA_API int32_t xetla_get_tensor_width_y(xetla_tdescriptor desc) {
    return desc.xetla_format<int32_t>().xetla_select<1, 1>(3)[0];
}

__XETLA_API int32_t xetla_get_tensor_pitch_x(xetla_tdescriptor desc) {
    return desc.xetla_format<int32_t>().xetla_select<1, 1>(4)[0];
}

__XETLA_API int32_t xetla_get_tensor_offset_x(xetla_tdescriptor desc) {
    return desc.xetla_format<int32_t>().xetla_select<1, 1>(5)[0];
}

__XETLA_API int32_t xetla_get_tensor_offset_y(xetla_tdescriptor desc) {
    return desc.xetla_format<int32_t>().xetla_select<1, 1>(6)[0];
}

__XETLA_API void xetla_set_block_widthx_widthy_arrlen(
        xetla_tdescriptor_ref desc, uint32_t block_widthx_widthy_arrlen) {
    desc.xetla_format<uint32_t>().xetla_select<1, 1>(7)
            = block_widthx_widthy_arrlen;
}

__XETLA_API void xetla_set_block_width_x(
        xetla_tdescriptor_ref desc, uint8_t width_x) {
    desc.xetla_format<uint8_t>().xetla_select<1, 1>(28) = width_x;
}

__XETLA_API void xetla_set_block_width_y(
        xetla_tdescriptor_ref desc, uint8_t width_y) {
    desc.xetla_format<uint8_t>().xetla_select<1, 1>(29) = width_y;
}

__XETLA_API void xetla_set_block_array_len(
        xetla_tdescriptor_ref desc, uint8_t array_len) {
    desc.xetla_format<uint8_t>().xetla_select<1, 1>(30) = array_len;
}

__XETLA_API void xetla_set_tensor_width_z(
        xetla_tdescriptor_ref desc, uint32_t width_z) {
    desc.xetla_format<uint32_t>().xetla_select<1, 1>(9) = width_z;
}

__XETLA_API void xetla_set_tensor_width_w(
        xetla_tdescriptor_ref desc, uint32_t width_w) {
    desc.xetla_format<uint32_t>().xetla_select<1, 1>(10) = width_w;
}

__XETLA_API void xetla_set_tensor_pitch_y(
        xetla_tdescriptor_ref desc, uint32_t pitch_y) {
    desc.xetla_format<uint32_t>().xetla_select<1, 1>(11) = pitch_y;
}

__XETLA_API void xetla_set_tensor_pitch_z(
        xetla_tdescriptor_ref desc, uint32_t pitch_z) {
    desc.xetla_format<uint32_t>().xetla_select<1, 1>(12) = pitch_z;
}

__XETLA_API void xetla_set_tensor_offset_z(
        xetla_tdescriptor_ref desc, int32_t offset_z) {
    desc.xetla_format<int32_t>().xetla_select<1, 1>(13) = offset_z;
}

__XETLA_API void xetla_set_tensor_offset_w(
        xetla_tdescriptor_ref desc, int32_t offset_w) {
    desc.xetla_format<int32_t>().xetla_select<1, 1>(14) = offset_w;
}

__XETLA_API void xetla_set_block_width_z(
        xetla_tdescriptor_ref desc, uint8_t width_z) {
    desc.xetla_format<uint8_t>().xetla_select<1, 1>(60) = width_z;
}

__XETLA_API void xetla_set_block_width_w(
        xetla_tdescriptor_ref desc, uint8_t width_w) {
    desc.xetla_format<uint8_t>().xetla_select<1, 1>(61) = width_w;
}
} // namespace detail

/// @addtogroup xetla_util_tensor_load_store
/// @{

/// @brief Tensor descriptor construction(global memory version).
/// Constructs a tensor descriptor based on the given arguments, check [here](https://gfxspecs.intel.com/Predator/Home/Index/63986) for more details.
/// @tparam Ty is the data type per element.
/// @tparam block_width is the width of the block to be loaded.
/// @tparam block_height is the height of the block to be loaded.
/// @tparam array_len is the array length of the block to be loaded.
/// @param tdesc [in|out] is the reference of tensor descriptor.
/// @param p [in] is the base address pointer of the tensor.
/// @param tensor_width [in] is the width of the tensor.
/// @param tensor_height [in] is the height of the tensor.
/// @param tensor_pitch [in] is the pitch(physical width of tensor in memory).
/// @param offset_x [in] is the x coordinate of the start point.
/// @param offset_y [in] is the y coordinate of the start point.
///
template <typename Ty, uint8_t block_width = 1, uint8_t block_height = 1,
        uint8_t array_len = 1>
__XETLA_API void xetla_fill_tdesc(xetla_tdescriptor_ref tdesc, Ty *p,
        int tensor_width, int tensor_height, int tensor_pitch, int offset_x,
        int offset_y) {
    detail::xetla_set_tensor_base_address(tdesc, (uint64_t)p);
    detail::xetla_set_tensor_width_x(tdesc, tensor_width * sizeof(Ty) - 1);
    detail::xetla_set_tensor_width_y(tdesc, tensor_height - 1);
    detail::xetla_set_tensor_pitch_x(tdesc, tensor_pitch * sizeof(Ty) - 1);
    detail::xetla_set_tensor_offset_x(tdesc, offset_x);
    detail::xetla_set_tensor_offset_y(tdesc, offset_y);
    uint32_t block_widthx_widthy_arrlen = (block_width - 1)
            | ((block_height - 1) << 8) | ((array_len - 1) << 16);
    detail::xetla_set_block_widthx_widthy_arrlen(
            tdesc, block_widthx_widthy_arrlen);
}

/// @brief Tensor descriptor construction(local memory version).
/// Constructs a tensor descriptor based on the given arguments, keep the same format as the global memory version.
/// @tparam Ty is the data type per element.
/// @param tdesc [in|out] is the reference of tensor descriptor.
/// @param base_address [in] is the local memory base address of the tensor.
/// @param tensor_width [in] is the width of the tensor.
/// @param tensor_height [in] is the height of the tensor.
/// @param tensor_pitch [in] is the pitch(physical width of tensor in memory).
/// @param offset_x [in] is the x coordinate of the start point.
/// @param offset_y [in] is the y coordinate of the start point.
///
template <typename Ty>
__XETLA_API void xetla_fill_tdesc(xetla_tdescriptor_ref tdesc,
        uint32_t base_address, int tensor_width, int tensor_height,
        int tensor_pitch, int offset_x, int offset_y) {
    detail::xetla_set_tensor_base_address(tdesc, base_address);
    detail::xetla_set_tensor_width_x(tdesc, tensor_width * sizeof(Ty));
    detail::xetla_set_tensor_width_y(tdesc, tensor_height);
    detail::xetla_set_tensor_pitch_x(tdesc, tensor_pitch * sizeof(Ty));
    detail::xetla_set_tensor_offset_x(tdesc, offset_x);
    detail::xetla_set_tensor_offset_y(tdesc, offset_y);
}

/// @brief Generate a new tensor descriptor(global memory version).
/// Generate a tensor descriptor based on the given arguments, check [here](https://gfxspecs.intel.com/Predator/Home/Index/63986) for more details.
/// @tparam Ty is the data type per element.
/// @tparam block_width is the width of the block to be loaded.
/// @tparam block_height is the height of the block to be loaded.
/// @tparam array_len is the array length of the block to be loaded.
/// @param p [in] is the base address pointer of the tensor.
/// @param tensor_width [in] is the width of the tensor.
/// @param tensor_height [in] is the height of the tensor.
/// @param tensor_pitch [in] is the pitch(physical width of tensor in memory).
/// @param offset_x [in] is the x coordinate of the start point.
/// @param offset_y [in] is the y coordinate of the start point.
/// @return return a new tensor
///
template <typename Ty, uint8_t block_width = 1, uint8_t block_height = 1,
        uint8_t array_len = 1>
__XETLA_API xetla_tdescriptor xetla_get_tdesc(Ty *p, int tensor_width,
        int tensor_height, int tensor_pitch, int offset_x, int offset_y) {
    xetla_tdescriptor tdesc;
    auto tdesc_ref = tdesc.xetla_format<uint32_t>();
    detail::xetla_set_tensor_base_address(tdesc_ref, (uint64_t)p);
    detail::xetla_set_tensor_width_x(tdesc_ref, tensor_width * sizeof(Ty) - 1);
    detail::xetla_set_tensor_width_y(tdesc_ref, tensor_height - 1);
    detail::xetla_set_tensor_pitch_x(tdesc_ref, tensor_pitch * sizeof(Ty) - 1);
    detail::xetla_set_tensor_offset_x(tdesc_ref, offset_x);
    detail::xetla_set_tensor_offset_y(tdesc_ref, offset_y);
    uint32_t block_widthx_widthy_arrlen = (block_width - 1)
            | ((block_height - 1) << 8) | ((array_len - 1) << 16);
    detail::xetla_set_block_widthx_widthy_arrlen(
            tdesc_ref, block_widthx_widthy_arrlen);
    return tdesc;
}

/// @brief Generate a new tensor descriptor(local memory version).
/// Constructs a tensor descriptor based on the given arguments, keep the same format as the global memory version.
/// @tparam Ty is the data type per element.
/// @param base_address [in] is the local memory base address of the tensor.
/// @param tensor_width [in] is the width of the tensor.
/// @param tensor_height [in] is the height of the tensor.
/// @param tensor_pitch [in] is the pitch(physical width of tensor in memory).
/// @param offset_x [in] is the x coordinate of the start point.
/// @param offset_y [in] is the y coordinate of the start point.
/// @return return a new tensor descriptor
///
template <typename Ty>
__XETLA_API xetla_tdescriptor xetla_get_tdesc(uint32_t base_address,
        int tensor_width, int tensor_height, int tensor_pitch, int offset_x,
        int offset_y) {
    xetla_tdescriptor tdesc;
    auto tdesc_ref = tdesc.xetla_format<uint32_t>();
    detail::xetla_set_tensor_base_address(tdesc_ref, base_address);
    detail::xetla_set_tensor_width_x(tdesc_ref, tensor_width * sizeof(Ty));
    detail::xetla_set_tensor_width_y(tdesc_ref, tensor_height);
    detail::xetla_set_tensor_pitch_x(tdesc_ref, tensor_pitch * sizeof(Ty));
    detail::xetla_set_tensor_offset_x(tdesc_ref, offset_x);
    detail::xetla_set_tensor_offset_y(tdesc_ref, offset_y);
    return tdesc;
}

/// @brief Update the x coordinate in the given tensor descriptor.
/// @param tdesc [in|our] is the reference of tensor descriptor.
/// @param doffset_x [in] is the offset (in number of data elements) in x direction.
__XETLA_API void xetla_update_tdesc_offsetx(
        xetla_tdescriptor_ref tdesc, int32_t doffset_x) {
    detail::xetla_set_tensor_offset_x(
            tdesc, detail::xetla_get_tensor_offset_x(tdesc) + doffset_x);
}

/// @brief Update the y coordinate in the given tensor descriptor.
/// @param tdesc [in|our] is the reference of tensor descriptor.
/// @param doffset_y [in] is the offset (in number of data elements) in y direction.
__XETLA_API void xetla_update_tdesc_offsety(
        xetla_tdescriptor_ref tdesc, int32_t doffset_y) {
    detail::xetla_set_tensor_offset_y(
            tdesc, detail::xetla_get_tensor_offset_y(tdesc) + doffset_y);
}

///
/// @brief Tensor load API.
/// This is tensor load API from global to registers. Check [here](https://gfxspecs.intel.com/Predator/Home/Index/53680) for more details.
/// @tparam Ty is the data type per element.
/// @tparam N is the total number of elements to load.
/// @tparam L1H is L1$ cache hint.
/// @tparam L3H is L3$ cache hint.
/// @tparam transpose is a flag to indicate whether the data is transposed during load.
/// @tparam transform is a flag to indicate whether the data is transformed (data pack inside dword) during load.
/// @param tdesc [in] is tensor descriptor including tensor base address, tensor dimensions, block size, etc.
/// @return xetla_vector is data returned from the load.
///
template <typename Ty, uint32_t N, cache_hint L1H = cache_hint::none,
        cache_hint L3H = cache_hint::none, bool transpose = false,
        bool transform = false>
__XETLA_API xetla_vector<Ty, N> xetla_tload_global(xetla_tdescriptor tdesc) {

    constexpr uint32_t numDst = 31 < ((N * sizeof(Ty) + 63) / 64)
            ? 31
            : ((N * sizeof(Ty) + 63) / 64);
    uint32_t msg_desc = 3;
    msg_desc |= (transform ? 1 : 0) << 7;
    msg_desc |= detail::get_element_size_code<sizeof(Ty)>() << 9;
    msg_desc |= (transpose ? 1 : 0) << 15;
    msg_desc |= detail::get_load_cache_hint_code<L1H, L3H>() << 17;
    msg_desc |= 1 << 25;
    msg_desc |= numDst << 20;

    constexpr uint32_t numSrc0 = 1;
    constexpr uint32_t execSize = 0;
    constexpr uint32_t sfid = 0xF;
    constexpr uint32_t exDesc = 0;

    xetla_vector<Ty, N> ret;
    xetla_raw_send<Ty, N, uint32_t, 16, execSize, sfid, numSrc0, numDst>(
            ret.xetla_format<native_type_t<Ty>>(), tdesc, exDesc, msg_desc);
    return ret;
}

///
/// @brief Tensor store API.
/// Tensor store API is to store a n-d (e.g. n=2) tensor into global using tensor descriptor. Check [here](https://gfxspecs.intel.com/Predator/Home/Index/53530) for more details.
/// @tparam Ty is the data type per element.
/// @tparam N is the number of elements to store.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param tdesc [in] is tensor descriptor including tensor base address, tensor dimensions, block size, etc.
/// @param data [in] is tensor data to store.
/// @return none.
///
template <typename Ty, uint32_t N, cache_hint L1H = cache_hint::none,
        cache_hint L3H = cache_hint::none>
__XETLA_API void xetla_tstore_global(
        xetla_tdescriptor tdesc, xetla_vector<Ty, N> data) {

    uint32_t msg_desc = 7; // store operation
    msg_desc |= detail::get_element_size_code<sizeof(Ty)>() << 9;
    msg_desc |= detail::get_store_cache_hint_code<L1H, L3H>() << 17;
    msg_desc |= 1 << 25;

    constexpr uint32_t numSrc1 = (N * sizeof(Ty) + 63) / 64;
    constexpr uint32_t numSrc0 = 1;
    constexpr uint32_t execSize = 0;
    constexpr uint32_t sfid = 0xF;
    constexpr uint32_t exDesc = 0;

    xetla_raw_send<uint32_t, 16, Ty, N, execSize, sfid, numSrc0, numSrc1>(
            tdesc, data, exDesc, msg_desc);
}

///
/// @brief Tensor prefetch API.
/// This is tensor prefetch API from global memory to L1$/L3$. Check [here](https://gfxspecs.intel.com/Predator/Home/Index/53680) for more details.
/// @tparam Ty is the data type per element.
/// @tparam L1H is L1$ cache hit.
/// @tparam L3H is L3$ cache hit.
/// @param tdesc is tensor descriptor including tensor base address, tensor dimensions, block size, etc.
/// @return none.
///
template <typename Ty, cache_hint L1H = cache_hint::cached,
        cache_hint L3H = cache_hint::cached>
__XETLA_API void xetla_tprefetch_global(xetla_tdescriptor tdesc) {

    uint32_t msg_desc = 3;
    msg_desc |= 0 << 7;
    msg_desc |= detail::get_element_size_code<sizeof(Ty)>() << 9;
    msg_desc |= 0 << 15;
    msg_desc |= detail::get_prefetch_cache_hint_code<L1H, L3H>() << 17;
    msg_desc |= 1 << 25;

    constexpr uint32_t numSrc0 = 1;
    constexpr uint32_t execSize = 0;
    constexpr uint32_t sfid = 0xF;
    constexpr uint32_t exDesc = 0;

    xetla_raw_send<uint32_t, 16, execSize, sfid, numSrc0>(
            tdesc, exDesc, msg_desc);
}

///
/// @brief Tensor atomic store API.
/// Tensor atomic store API is to store a n-d (e.g. n=2) tensor into global. Check [here](https://gfxspecs.intel.com/Predator/Home/Index/53548) for more details.
/// @tparam Ty is the data type per element.
/// @tparam N is the number of elements to store.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param address [in] is is the 64bit address for each channel.
/// @param data [in] is tensor data to store.
/// @return none.
///
template <typename Ty, uint32_t N, cache_hint L1H = cache_hint::none,
        cache_hint L3H = cache_hint::none, atomic_op Op>
__XETLA_API void xetla_tatomic_store_global(xetla_vector<uint64_t, N> address,
        xetla_vector<Ty, N> data, xetla_mask<N> pred = 1) {

    constexpr uint32_t numSrc0 = (N * sizeof(uint64_t) + 63) / 64;
    constexpr uint32_t numSrc1 = (N * sizeof(Ty) + 63) / 64;
    constexpr uint32_t num_dest = (N * sizeof(Ty) + 63) / 64;

    uint32_t msg_desc = detail::get_atomic_opcode<Op>();
    ///only support 64bit address
    msg_desc |= 3 << 7;
    msg_desc |= detail::get_element_size_code<sizeof(Ty)>() << 9;
    msg_desc |= detail::get_atomic_cache_hint_code<L1H, L3H>() << 17;
    msg_desc |= numSrc0 << 25;

    constexpr uint32_t execSize = detail::get_execSize_code<N>();
    constexpr uint32_t sfid = 0xF;
    constexpr uint32_t exDesc = 0;

    xetla_raw_send<uint64_t, N, Ty, N, execSize, sfid, numSrc0, numSrc1>(
            address, data, exDesc, msg_desc, pred);
}

/// @} xetla_util_tensor_load_store

} // namespace gpu::xetla
