/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 18, 2025.
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

#ifndef _CUDA_STD___TUPLE_TUPLE_SIZE_H
#define _CUDA_STD___TUPLE_TUPLE_SIZE_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__fwd/tuple.h>
#include <uscl/std/__tuple_dir/tuple_types.h>
#include <uscl/std/__type_traits/enable_if.h>
#include <uscl/std/__type_traits/integral_constant.h>
#include <uscl/std/__type_traits/is_const.h>
#include <uscl/std/__type_traits/is_volatile.h>
#include <uscl/std/cstddef>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_size;

template <class _Tp, class...>
using __enable_if_tuple_size_imp = _Tp;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
tuple_size<__enable_if_tuple_size_imp<const _Tp,
                                      enable_if_t<!is_volatile<_Tp>::value>,
                                      integral_constant<size_t, sizeof(tuple_size<_Tp>)>>>
    : public integral_constant<size_t, tuple_size<_Tp>::value>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
tuple_size<__enable_if_tuple_size_imp<volatile _Tp,
                                      enable_if_t<!is_const<_Tp>::value>,
                                      integral_constant<size_t, sizeof(tuple_size<_Tp>)>>>
    : public integral_constant<size_t, tuple_size<_Tp>::value>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
tuple_size<__enable_if_tuple_size_imp<const volatile _Tp, integral_constant<size_t, sizeof(tuple_size<_Tp>)>>>
    : public integral_constant<size_t, tuple_size<_Tp>::value>
{};

template <class... _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_size<tuple<_Tp...>> : public integral_constant<size_t, sizeof...(_Tp)>
{};

template <class... _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
tuple_size<__tuple_types<_Tp...>> : public integral_constant<size_t, sizeof...(_Tp)>
{};

template <class _Tp>
inline constexpr size_t tuple_size_v = tuple_size<_Tp>::value;

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TUPLE_SIZE_H
