/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 23, 2022.
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

#ifndef _CUDA_STD___TYPE_TRAITS_IS_CORE_CONVERTIBLE_H
#define _CUDA_STD___TYPE_TRAITS_IS_CORE_CONVERTIBLE_H

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

// [conv.general]/3 says "E is convertible to T" whenever "T t=E;" is well-formed.
// We can't test for that, but we can test implicit convertibility by passing it
// to a function. Notice that __is_core_convertible<void,void> is false,
// and __is_core_convertible<immovable-type,immovable-type> is true in C++17 and later.

template <class _Tp, class _Up, class = void>
struct __is_core_convertible : public false_type
{};

template <class _Tp, class _Up>
struct __is_core_convertible<_Tp, _Up, decltype(static_cast<void (*)(_Up)>(0)(static_cast<_Tp (*)()>(0)()))>
    : public true_type
{};

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_CORE_CONVERTIBLE_H
