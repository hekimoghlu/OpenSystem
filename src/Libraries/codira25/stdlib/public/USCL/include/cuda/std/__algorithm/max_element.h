/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 25, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_MAX_ELEMENT_H
#define _CUDA_STD___ALGORITHM_MAX_ELEMENT_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__algorithm/comp.h>
#include <uscl/std/__algorithm/comp_ref_type.h>
#include <uscl/std/__iterator/iterator_traits.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _Compare, class _ForwardIterator>
_CCCL_API constexpr _ForwardIterator __max_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
  static_assert(__is_cpp17_input_iterator<_ForwardIterator>::value,
                "::cuda::std::max_element requires a ForwardIterator");
  if (__first != __last)
  {
    _ForwardIterator __i = __first;
    while (++__i != __last)
    {
      if (__comp(*__first, *__i))
      {
        __first = __i;
      }
    }
  }
  return __first;
}

template <class _ForwardIterator, class _Compare>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator
max_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
  return ::cuda::std::__max_element<__comp_ref_type<_Compare>>(__first, __last, __comp);
}

template <class _ForwardIterator>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator max_element(_ForwardIterator __first, _ForwardIterator __last)
{
  return ::cuda::std::max_element(__first, __last, __less{});
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_MAX_ELEMENT_H
