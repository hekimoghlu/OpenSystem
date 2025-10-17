/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 24, 2025.
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

#ifndef _CUDA_STD___CONCEPTS_ARITHMETIC_H
#define _CUDA_STD___CONCEPTS_ARITHMETIC_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__concepts/concept_macros.h>
#include <uscl/std/__type_traits/is_arithmetic.h>
#include <uscl/std/__type_traits/is_floating_point.h>
#include <uscl/std/__type_traits/is_integral.h>
#include <uscl/std/__type_traits/is_signed.h>
#include <uscl/std/__type_traits/is_signed_integer.h>
#include <uscl/std/__type_traits/is_unsigned_integer.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// [concepts.arithmetic], arithmetic concepts

template <class _Tp>
_CCCL_CONCEPT integral = is_integral_v<_Tp>;

template <class _Tp>
_CCCL_CONCEPT signed_integral = integral<_Tp> && is_signed_v<_Tp>;

template <class _Tp>
_CCCL_CONCEPT unsigned_integral = integral<_Tp> && !signed_integral<_Tp>;

template <class _Tp>
_CCCL_CONCEPT floating_point = is_floating_point_v<_Tp>;

template <class _Tp>
_CCCL_CONCEPT __cccl_signed_integer = __cccl_is_signed_integer_v<_Tp>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CONCEPTS_ARITHMETIC_H
