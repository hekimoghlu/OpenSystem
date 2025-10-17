/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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
#ifndef _CUDA___BARRIER_BARRIER_THREAD_SCOPE_H
#define _CUDA___BARRIER_BARRIER_THREAD_SCOPE_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/__barrier/barrier_block_scope.h>
#include <uscl/__fwd/barrier.h>
#include <uscl/std/__atomic/scopes.h>
#include <uscl/std/__barrier/empty_completion.h>
#include <uscl/std/cstdint>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <>
class barrier<thread_scope_thread, ::cuda::std::__empty_completion> : private barrier<thread_scope_block>
{
  using __base = barrier<thread_scope_block>;

public:
  using __base::__base;

  _CCCL_API inline friend void init(barrier* __b,
                                    ::cuda::std::ptrdiff_t __expected,
                                    ::cuda::std::__empty_completion __completion = ::cuda::std::__empty_completion())
  {
    init(static_cast<__base*>(__b), __expected, __completion);
  }

  using __base::arrive;
  using __base::arrive_and_drop;
  using __base::arrive_and_wait;
  using __base::max;
  using __base::wait;
};

_CCCL_END_NAMESPACE_CUDA

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA___BARRIER_BARRIER_THREAD_SCOPE_H
