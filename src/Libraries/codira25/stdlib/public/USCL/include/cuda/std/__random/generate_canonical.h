/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 3, 2023.
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

#ifndef _CUDA_STD___RANDOM_GENERATE_CANONICAL_H
#define _CUDA_STD___RANDOM_GENERATE_CANONICAL_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__bit/integral.h>
#include <uscl/std/cstdint>
#include <uscl/std/limits>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// generate_canonical
_CCCL_EXEC_CHECK_DISABLE
template <class _RealType, size_t __bits, class _URng>
[[nodiscard]] _CCCL_API _RealType generate_canonical(_URng& __g) noexcept
{
  constexpr size_t __dt = numeric_limits<_RealType>::digits;
  const size_t __b      = __dt < __bits ? __dt : __bits;
  const size_t __log_r  = ::cuda::std::__bit_log2<uint64_t>((_URng::max) () - (_URng::min) () + uint64_t(1));
  const size_t __k      = __b / __log_r + (__b % __log_r != 0) + (__b == 0);
  const _RealType __rp  = static_cast<_RealType>((_URng::max) () - (_URng::min) ()) + _RealType(1);
  _RealType __base      = __rp;
  _RealType __sp        = __g() - (_URng::min) ();

  _CCCL_PRAGMA_UNROLL_FULL()
  for (size_t __i = 1; __i < __k; ++__i, __base *= __rp)
  {
    __sp += (__g() - (_URng::min) ()) * __base;
  }
  return __sp / __base;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANDOM_GENERATE_CANONICAL_H
