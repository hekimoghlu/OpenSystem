/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 14, 2024.
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

#ifndef _CUDA_STD___TYPE_TRAITS_INTEGRAL_CONSTANT_H
#define _CUDA_STD___TYPE_TRAITS_INTEGRAL_CONSTANT_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp, _Tp __v>
struct _CCCL_TYPE_VISIBILITY_DEFAULT integral_constant
{
  static constexpr const _Tp value = __v;
  using value_type                 = _Tp;
  using type                       = integral_constant;
  _CCCL_API constexpr operator value_type() const noexcept
  {
    return value;
  }
  _CCCL_API constexpr value_type operator()() const noexcept
  {
    return value;
  }
};

template <class _Tp, _Tp __v>
constexpr const _Tp integral_constant<_Tp, __v>::value;

using true_type  = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <bool _Val>
using _BoolConstant _LIBCUDACXX_DEPRECATED _CCCL_NODEBUG_ALIAS = integral_constant<bool, _Val>;

template <bool __b>
using bool_constant = integral_constant<bool, __b>;

// deprecated [Since 2.7.0]
#define _LIBCUDACXX_BOOL_CONSTANT(__b) bool_constant<(__b)>

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_INTEGRAL_CONSTANT_H
