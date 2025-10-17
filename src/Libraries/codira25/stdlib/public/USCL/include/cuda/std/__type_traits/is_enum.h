/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 26, 2023.
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

#ifndef _CUDA_STD___TYPE_TRAITS_IS_ENUM_H
#define _CUDA_STD___TYPE_TRAITS_IS_ENUM_H

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
#include <uscl/std/__type_traits/is_floating_point.h>
#include <uscl/std/__type_traits/is_function.h>
#include <uscl/std/__type_traits/is_integral.h>
#include <uscl/std/__type_traits/is_member_pointer.h>
#include <uscl/std/__type_traits/is_pointer.h>
#include <uscl/std/__type_traits/is_reference.h>
#include <uscl/std/__type_traits/is_union.h>
#include <uscl/std/__type_traits/is_void.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_IS_ENUM) && !defined(_LIBCUDACXX_USE_IS_ENUM_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_enum : public integral_constant<bool, _CCCL_BUILTIN_IS_ENUM(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_enum_v = _CCCL_BUILTIN_IS_ENUM(_Tp);

#else

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_enum
    : public integral_constant<
        bool,
        !is_void<_Tp>::value && !is_integral<_Tp>::value && !is_floating_point<_Tp>::value && !is_array<_Tp>::value
          && !is_pointer<_Tp>::value && !is_reference<_Tp>::value && !is_member_pointer<_Tp>::value
          && !is_union<_Tp>::value && !is_class<_Tp>::value && !is_function<_Tp>::value>
{};

template <class _Tp>
inline constexpr bool is_enum_v = is_enum<_Tp>::value;

#endif // defined(_CCCL_BUILTIN_IS_ENUM) && !defined(_LIBCUDACXX_USE_IS_ENUM_FALLBACK)

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_ENUM_H
