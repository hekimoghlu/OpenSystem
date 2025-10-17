/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 2, 2022.
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

#ifndef _CUDA_STD___TYPE_TRAITS_IS_UNSIGNED_INTEGER_H
#define _CUDA_STD___TYPE_TRAITS_IS_UNSIGNED_INTEGER_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__type_traits/remove_cv.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
inline constexpr bool __cccl_is_unsigned_integer_v = false;

template <>
inline constexpr bool __cccl_is_unsigned_integer_v<unsigned char> = true;

template <>
inline constexpr bool __cccl_is_unsigned_integer_v<unsigned short> = true;

template <>
inline constexpr bool __cccl_is_unsigned_integer_v<unsigned int> = true;

template <>
inline constexpr bool __cccl_is_unsigned_integer_v<unsigned long> = true;

template <>
inline constexpr bool __cccl_is_unsigned_integer_v<unsigned long long> = true;

#if _CCCL_HAS_INT128()
template <>
inline constexpr bool __cccl_is_unsigned_integer_v<__uint128_t> = true;
#endif // _CCCL_HAS_INT128()

template <class _Tp>
inline constexpr bool __cccl_is_cv_unsigned_integer_v = __cccl_is_unsigned_integer_v<remove_cv_t<_Tp>>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_UNSIGNED_INTEGER_H
