/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 19, 2022.
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

#ifndef _CUDA_STD___TYPE_TRAITS_IS_OBJECT_H
#define _CUDA_STD___TYPE_TRAITS_IS_OBJECT_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__type_traits/integral_constant.h>
#include <uscl/std/__type_traits/is_array.h>
#include <uscl/std/__type_traits/is_class.h>
#include <uscl/std/__type_traits/is_scalar.h>
#include <uscl/std/__type_traits/is_union.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_IS_OBJECT) && !defined(_LIBCUDACXX_USE_IS_OBJECT_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_object : public bool_constant<_CCCL_BUILTIN_IS_OBJECT(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_object_v = _CCCL_BUILTIN_IS_OBJECT(_Tp);

#else

template <class _Tp>
inline constexpr bool is_object_v = is_scalar_v<_Tp> || is_array_v<_Tp> || is_union_v<_Tp> || is_class_v<_Tp>;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_object : public bool_constant<is_object_v<_Tp>>
{};

#endif // defined(_CCCL_BUILTIN_IS_OBJECT) && !defined(_LIBCUDACXX_USE_IS_OBJECT_FALLBACK)

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_OBJECT_H
