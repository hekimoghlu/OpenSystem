/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 2, 2022.
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

#ifndef _CUDA_STD___ALGORITHM_SHIFT_LEFT_H
#define _CUDA_STD___ALGORITHM_SHIFT_LEFT_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__algorithm/move.h>
#include <uscl/std/__iterator/iterator_traits.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _ForwardIterator>
_CCCL_API constexpr _ForwardIterator __shift_left(
  _ForwardIterator __first,
  _ForwardIterator __last,
  typename iterator_traits<_ForwardIterator>::difference_type __n,
  random_access_iterator_tag)
{
  _ForwardIterator __m = __first;
  if (__n >= __last - __first)
  {
    return __first;
  }
  __m += __n;
  return ::cuda::std::move(__m, __last, __first);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _ForwardIterator>
_CCCL_API constexpr _ForwardIterator __shift_left(
  _ForwardIterator __first,
  _ForwardIterator __last,
  typename iterator_traits<_ForwardIterator>::difference_type __n,
  forward_iterator_tag)
{
  _ForwardIterator __m = __first;
  for (; __n > 0; --__n)
  {
    if (__m == __last)
    {
      return __first;
    }
    ++__m;
  }
  return ::cuda::std::move(__m, __last, __first);
}

template <class _ForwardIterator>
_CCCL_API constexpr _ForwardIterator shift_left(
  _ForwardIterator __first, _ForwardIterator __last, typename iterator_traits<_ForwardIterator>::difference_type __n)
{
  if (__n == 0)
  {
    return __last;
  }

  using _IterCategory = typename iterator_traits<_ForwardIterator>::iterator_category;
  return __shift_left(__first, __last, __n, _IterCategory());
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_SHIFT_LEFT_H
