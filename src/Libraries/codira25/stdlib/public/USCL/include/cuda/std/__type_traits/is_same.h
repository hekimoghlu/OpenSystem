/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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

#ifndef _CUDA_STD___TYPE_TRAITS_IS_SAME_H
#define _CUDA_STD___TYPE_TRAITS_IS_SAME_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__type_traits/integral_constant.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_IS_SAME) && !defined(_LIBCUDACXX_USE_IS_SAME_FALLBACK)

template <class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_same : bool_constant<_CCCL_BUILTIN_IS_SAME(_Tp, _Up)>
{};

template <class _Tp, class _Up>
inline constexpr bool is_same_v = _CCCL_BUILTIN_IS_SAME(_Tp, _Up);

// _IsSame<T,U> has the same effect as is_same<T,U> but instantiates fewer types:
// is_same<A,B> and is_same<C,D> are guaranteed to be different types, but
// _IsSame<A,B> and _IsSame<C,D> are the same type (namely, false_type).
// Neither GCC nor Clang can mangle the __is_same builtin, so _IsSame
// mustn't be directly used anywhere that contributes to name-mangling
// (such as in a dependent return type).

template <class _Tp, class _Up>
using _IsSame = bool_constant<_CCCL_BUILTIN_IS_SAME(_Tp, _Up)>;

template <class _Tp, class _Up>
using _IsNotSame = bool_constant<!_CCCL_BUILTIN_IS_SAME(_Tp, _Up)>;

#else

template <class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_same : public false_type
{};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_same<_Tp, _Tp> : public true_type
{};

template <class _Tp, class _Up>
inline constexpr bool is_same_v = false;
template <class _Tp>
inline constexpr bool is_same_v<_Tp, _Tp> = true;

// _IsSame<T,U> has the same effect as is_same<T,U> but instantiates fewer types:
// is_same<A,B> and is_same<C,D> are guaranteed to be different types, but
// _IsSame<A,B> and _IsSame<C,D> are the same type (namely, false_type).
// Neither GCC nor Clang can mangle the __is_same builtin, so _IsSame
// mustn't be directly used anywhere that contributes to name-mangling
// (such as in a dependent return type).

template <class _Tp, class _Up>
using _IsSame = bool_constant<is_same_v<_Tp, _Up>>;

template <class _Tp, class _Up>
using _IsNotSame = bool_constant<!is_same_v<_Tp, _Up>>;

#endif // defined(_CCCL_BUILTIN_IS_SAME) && !defined(_LIBCUDACXX_USE_IS_SAME_FALLBACK)

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_SAME_H
