/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 28, 2024.
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
#ifndef _CUDA_STD___NEW_BAD_ALLOC_H
#define _CUDA_STD___NEW_BAD_ALLOC_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__exception/terminate.h>

#if _CCCL_HAS_EXCEPTIONS()
#  include <new>
#endif // !_CCCL_HAS_EXCEPTIONS()

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

[[noreturn]] _CCCL_API inline void __throw_bad_alloc()
{
#if _CCCL_HAS_EXCEPTIONS()
  NV_IF_ELSE_TARGET(NV_IS_HOST, (throw ::std::bad_alloc();), (::cuda::std::terminate();))
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
  ::cuda::std::terminate();
#endif // !_CCCL_HAS_EXCEPTIONS()
}

[[noreturn]] _CCCL_API inline void __throw_bad_array_new_length()
{
#if _CCCL_HAS_EXCEPTIONS()
  NV_IF_ELSE_TARGET(NV_IS_HOST, (throw ::std::bad_array_new_length();), (::cuda::std::terminate();))
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
  ::cuda::std::terminate();
#endif // !_CCCL_HAS_EXCEPTIONS()
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___NEW_BAD_ALLOC_H
