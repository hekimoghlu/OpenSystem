/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 3, 2023.
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

#ifndef _CUDA_STD___TYPE_TRAITS_IS_NOTHROW_DESTRUCTIBLE_H
#define _CUDA_STD___TYPE_TRAITS_IS_NOTHROW_DESTRUCTIBLE_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__type_traits/integral_constant.h>
#include <uscl/std/__type_traits/is_destructible.h>
#include <uscl/std/__type_traits/is_reference.h>
#include <uscl/std/__type_traits/is_scalar.h>
#include <uscl/std/__type_traits/remove_all_extents.h>
#include <uscl/std/__utility/declval.h>
#include <uscl/std/cstddef>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// is_nothrow_destructible

#if defined(_CCCL_BUILTIN_IS_NOTHROW_DESTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_NOTHROW_DESTRUCTIBLE_FALLBACK)

template <class _Tp>
struct is_nothrow_destructible : public integral_constant<bool, _CCCL_BUILTIN_IS_NOTHROW_DESTRUCTIBLE(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_nothrow_destructible_v = _CCCL_BUILTIN_IS_NOTHROW_DESTRUCTIBLE(_Tp);

#else // ^^^ _CCCL_BUILTIN_IS_NOTHROW_DESTRUCTIBLE ^^^ / vvv !_CCCL_BUILTIN_IS_NOTHROW_DESTRUCTIBLE vvv

template <class _Tp, bool = is_destructible<_Tp>::value>
struct __cccl_is_nothrow_destructible : false_type
{};

template <class _Tp>
struct __cccl_is_nothrow_destructible<_Tp, true>
    : public integral_constant<bool, noexcept(::cuda::std::declval<_Tp>().~_Tp())>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_destructible : public __cccl_is_nothrow_destructible<_Tp>
{};

template <class _Tp, size_t _Ns>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_destructible<_Tp[_Ns]> : public is_nothrow_destructible<_Tp>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_destructible<_Tp&> : public true_type
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_destructible<_Tp&&> : public true_type
{};

template <class _Tp>
inline constexpr bool is_nothrow_destructible_v = is_nothrow_destructible<_Tp>::value;

#endif // !_CCCL_BUILTIN_IS_NOTHROW_DESTRUCTIBLE

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_NOTHROW_DESTRUCTIBLE_H
