/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 25, 2024.
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
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_EXTENT_H
#define _CUDA_STD___TYPE_TRAITS_EXTENT_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__type_traits/integral_constant.h>
#include <uscl/std/cstddef>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_ARRAY_EXTENT) && !defined(_LIBCUDACXX_USE_ARRAY_EXTENT_FALLBACK)

template <class _Tp, size_t _Dim = 0>
struct _CCCL_TYPE_VISIBILITY_DEFAULT extent : integral_constant<size_t, _CCCL_BUILTIN_ARRAY_EXTENT(_Tp, _Dim)>
{};

template <class _Tp, unsigned _Ip = 0>
inline constexpr size_t extent_v = _CCCL_BUILTIN_ARRAY_EXTENT(_Tp, _Ip);

#else // ^^^ _CCCL_BUILTIN_ARRAY_EXTENT ^^^ / vvv !_CCCL_BUILTIN_ARRAY_EXTENT vvv

template <class _Tp, unsigned _Ip = 0>
struct _CCCL_TYPE_VISIBILITY_DEFAULT extent : public integral_constant<size_t, 0>
{};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT extent<_Tp[], 0> : public integral_constant<size_t, 0>
{};
template <class _Tp, unsigned _Ip>
struct _CCCL_TYPE_VISIBILITY_DEFAULT extent<_Tp[], _Ip> : public integral_constant<size_t, extent<_Tp, _Ip - 1>::value>
{};
template <class _Tp, size_t _Np>
struct _CCCL_TYPE_VISIBILITY_DEFAULT extent<_Tp[_Np], 0> : public integral_constant<size_t, _Np>
{};
template <class _Tp, size_t _Np, unsigned _Ip>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
extent<_Tp[_Np], _Ip> : public integral_constant<size_t, extent<_Tp, _Ip - 1>::value>
{};

template <class _Tp, unsigned _Ip = 0>
inline constexpr size_t extent_v = extent<_Tp, _Ip>::value;

#endif // !_CCCL_BUILTIN_ARRAY_EXTENT

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_EXTENT_H
