/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 24, 2021.
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
#ifndef _CUDA_STD___CSTDDEF_TYPES_H
#define _CUDA_STD___CSTDDEF_TYPES_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)
#  include <cstddef>
#else
#  if !defined(offsetof)
#    define offsetof(type, member) (::size_t) ((char*) &(((type*) 0)->member) - (char*) 0)
#  endif // !offsetof
#endif // !_CCCL_COMPILER(NVRTC)

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_COMPILER(NVRTC)
using max_align_t = long double;
#else // ^^^ _CCCL_COMPILER(NVRTC) ^^^ / vvv !_CCCL_COMPILER(NVRTC) vvv
// Re-use the compiler's <stddef.h> max_align_t where possible.
using ::max_align_t;
#endif // _CCCL_COMPILER(NVRTC)

using nullptr_t = decltype(nullptr);
using ::ptrdiff_t;
using ::size_t;

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CSTDDEF_TYPES_H
