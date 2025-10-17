/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 26, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___UTILITY_IN_PLACE_H
#define _CUDA_STD___UTILITY_IN_PLACE_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__type_traits/is_reference.h>
#include <uscl/std/__type_traits/remove_cvref.h>
#include <uscl/std/__type_traits/remove_reference.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

struct _CCCL_TYPE_VISIBILITY_DEFAULT in_place_t
{
  _CCCL_HIDE_FROM_ABI explicit in_place_t() = default;
};
_CCCL_GLOBAL_CONSTANT in_place_t in_place{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT in_place_type_t
{
  _CCCL_HIDE_FROM_ABI explicit in_place_type_t() = default;
};
template <class _Tp>
inline constexpr in_place_type_t<_Tp> in_place_type{};

template <size_t _Idx>
struct _CCCL_TYPE_VISIBILITY_DEFAULT in_place_index_t
{
  _CCCL_HIDE_FROM_ABI explicit in_place_index_t() = default;
};
template <size_t _Idx>
inline constexpr in_place_index_t<_Idx> in_place_index{};

template <class _Tp>
struct __is_inplace_type_imp : false_type
{};
template <class _Tp>
struct __is_inplace_type_imp<in_place_type_t<_Tp>> : true_type
{};

template <class _Tp>
using __is_inplace_type = __is_inplace_type_imp<remove_cvref_t<_Tp>>;

template <class _Tp>
struct __is_inplace_index_imp : false_type
{};
template <size_t _Idx>
struct __is_inplace_index_imp<in_place_index_t<_Idx>> : true_type
{};

template <class _Tp>
using __is_inplace_index = __is_inplace_index_imp<remove_cvref_t<_Tp>>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___UTILITY_IN_PLACE_H
