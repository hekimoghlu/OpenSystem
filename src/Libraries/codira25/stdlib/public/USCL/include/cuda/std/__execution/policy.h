/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___EXECUTION_POLICY_H
#define _CUDA_STD___EXECUTION_POLICY_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__type_traits/underlying_type.h>
#include <uscl/std/cstdint>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_EXECUTION

enum class __execution_policy : uint32_t
{
  __invalid_execution_policy = 0,
  __sequenced                = 1 << 0,
  __parallel                 = 1 << 1,
  __unsequenced              = 1 << 2,
  __parallel_unsequenced     = __execution_policy::__parallel | __execution_policy::__unsequenced,
};

[[nodiscard]] _CCCL_API constexpr bool
__satisfies_execution_policy(__execution_policy __lhs, __execution_policy __rhs) noexcept
{
  return (static_cast<uint32_t>(__lhs) & static_cast<uint32_t>(__rhs)) != 0;
}

template <__execution_policy _Policy>
struct __policy
{
  template <__execution_policy _OtherPolicy>
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const __policy&, const __policy<_OtherPolicy>&) noexcept
  {
    using __underlying_t = underlying_type_t<__execution_policy>;
    return (static_cast<__underlying_t>(_Policy) == static_cast<__underlying_t>(_OtherPolicy));
  }

#if _CCCL_STD_VER <= 2017
  template <__execution_policy _OtherPolicy>
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const __policy&, const __policy<_OtherPolicy>&) noexcept
  {
    using __underlying_t = underlying_type_t<__execution_policy>;
    return (static_cast<__underlying_t>(_Policy) != static_cast<__underlying_t>(_OtherPolicy));
  }
#endif // _CCCL_STD_VER <= 2017

  static constexpr __execution_policy __policy_ = _Policy;
};

struct sequenced_policy : public __policy<__execution_policy::__sequenced>
{};

_CCCL_GLOBAL_CONSTANT sequenced_policy seq{};

struct parallel_policy : public __policy<__execution_policy::__parallel>
{};
_CCCL_GLOBAL_CONSTANT parallel_policy par{};

struct parallel_unsequenced_policy : public __policy<__execution_policy::__parallel_unsequenced>
{};
_CCCL_GLOBAL_CONSTANT parallel_unsequenced_policy par_unseq{};

struct unsequenced_policy : public __policy<__execution_policy::__unsequenced>
{};
_CCCL_GLOBAL_CONSTANT unsequenced_policy unseq{};

_CCCL_END_NAMESPACE_EXECUTION

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___EXECUTION_POLICY_H
