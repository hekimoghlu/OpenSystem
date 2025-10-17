/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 4, 2021.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_REFERENCE_CONSTRUCTS_FROM_TEMPORARY_H
#define _CUDA_STD___TYPE_TRAITS_REFERENCE_CONSTRUCTS_FROM_TEMPORARY_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__type_traits/always_false.h>
#include <uscl/std/__type_traits/integral_constant.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY)
template <class _Tp, class _Up>
struct reference_constructs_from_temporary
    : integral_constant<bool, _CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY(_Tp, _Up)>
{};

template <class _Tp, class _Up>
inline constexpr bool reference_constructs_from_temporary_v =
  _CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY(_Tp, _Up);

#else

template <class _Tp, class _Up>
struct reference_constructs_from_temporary : integral_constant<bool, false>
{
  static_assert(__always_false_v<_Tp>, "The compiler does not support __reference_constructs_from_temporary");
};

template <class _Tp, class _Up>
inline constexpr bool reference_constructs_from_temporary_v = reference_constructs_from_temporary<_Tp, _Up>::value;

#endif // !_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_REFERENCE_CONSTRUCTS_FROM_TEMPORARY_H
