/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 12, 2022.
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

#ifndef _CUDA_STD___ALGORITHM_INCLUDES_H
#define _CUDA_STD___ALGORITHM_INCLUDES_H

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
#include <uscl/std/__functional/identity.h>
#include <uscl/std/__functional/invoke.h>
#include <uscl/std/__iterator/iterator_traits.h>
#include <uscl/std/__type_traits/is_callable.h>
#include <uscl/std/__utility/move.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _Iter1, class _Sent1, class _Iter2, class _Sent2, class _Comp, class _Proj1, class _Proj2>
_CCCL_API constexpr bool __includes(
  _Iter1 __first1, _Sent1 __last1, _Iter2 __first2, _Sent2 __last2, _Comp&& __comp, _Proj1&& __proj1, _Proj2&& __proj2)
{
  for (; __first2 != __last2; ++__first1)
  {
    if (__first1 == __last1
        || ::cuda::std::__invoke(
          __comp, ::cuda::std::__invoke(__proj2, *__first2), ::cuda::std::__invoke(__proj1, *__first1)))
    {
      return false;
    }
    if (!::cuda::std::__invoke(
          __comp, ::cuda::std::__invoke(__proj1, *__first1), ::cuda::std::__invoke(__proj2, *__first2)))
    {
      ++__first2;
    }
  }
  return true;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _InputIterator1, class _InputIterator2, class _Compare>
[[nodiscard]] _CCCL_API constexpr bool includes(
  _InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2, _Compare __comp)
{
  static_assert(__is_callable<_Compare, decltype(*__first1), decltype(*__first2)>::value,
                "Comparator has to be callable");

  return ::cuda::std::__includes(
    ::cuda::std::move(__first1),
    ::cuda::std::move(__last1),
    ::cuda::std::move(__first2),
    ::cuda::std::move(__last2),
    static_cast<__comp_ref_type<_Compare>>(__comp),
    identity(),
    identity());
}

_CCCL_EXEC_CHECK_DISABLE
template <class _InputIterator1, class _InputIterator2>
[[nodiscard]] _CCCL_API constexpr bool
includes(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2)
{
  return ::cuda::std::includes(
    ::cuda::std::move(__first1),
    ::cuda::std::move(__last1),
    ::cuda::std::move(__first2),
    ::cuda::std::move(__last2),
    __less{});
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_INCLUDES_H
