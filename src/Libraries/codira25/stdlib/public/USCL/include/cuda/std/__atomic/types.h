/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA_STD___ATOMIC_TYPES_H
#define __CUDA_STD___ATOMIC_TYPES_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__atomic/types/base.h>
#include <uscl/std/__atomic/types/locked.h>
#include <uscl/std/__atomic/types/reference.h>
#include <uscl/std/__atomic/types/small.h>
#include <uscl/std/__type_traits/conditional.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <typename _Tp>
struct __atomic_traits
{
  static constexpr bool __atomic_requires_lock      = !__atomic_is_always_lock_free<_Tp>::__value;
  static constexpr bool __atomic_requires_small     = sizeof(_Tp) < 4;
  static constexpr bool __atomic_supports_reference = __atomic_is_always_lock_free<_Tp>::__value && sizeof(_Tp) <= 8;
};

template <typename _Tp>
using __atomic_storage_t =
  _If<__atomic_traits<_Tp>::__atomic_requires_small,
      __atomic_small_storage<_Tp>,
      _If<__atomic_traits<_Tp>::__atomic_requires_lock, __atomic_locked_storage<_Tp>, __atomic_storage<_Tp>>>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // __CUDA_STD___ATOMIC_TYPES_H
