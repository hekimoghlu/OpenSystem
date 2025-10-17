/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 23, 2022.
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

// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CSTDLIB_MALLOC_H
#define _CUDA_STD___CSTDLIB_MALLOC_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__cstddef/types.h>
#include <uscl/std/cstring>

#if !_CCCL_COMPILER(NVRTC)
#  include <cstdlib>
#endif // !_CCCL_COMPILER(NVRTC)

#include <nv/target>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

using ::free;
using ::malloc;

#if _CCCL_CUDA_COMPILATION()
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE void* __calloc_device(size_t __n, size_t __size) noexcept
{
  void* __ptr{};

  const size_t __nbytes = __n * __size;

  if (::__umul64hi(__n, __size) == 0)
  {
    __ptr = ::cuda::std::malloc(__nbytes);
    if (__ptr != nullptr)
    {
      ::cuda::std::memset(__ptr, 0, __nbytes);
    }
  }

  return __ptr;
}
#endif // _CCCL_CUDA_COMPILATION()

[[nodiscard]] _CCCL_API inline void* calloc(size_t __n, size_t __size) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (return ::calloc(__n, __size);), (return ::cuda::std::__calloc_device(__n, __size);))
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CSTDLIB_MALLOC_H
