/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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
#ifndef _CUDA_STD___NUMERIC_PARTIAL_SUM_H
#define _CUDA_STD___NUMERIC_PARTIAL_SUM_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__iterator/iterator_traits.h>
#include <uscl/std/__utility/move.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _InputIterator, class _OutputIterator>
_CCCL_API constexpr _OutputIterator partial_sum(_InputIterator __first, _InputIterator __last, _OutputIterator __result)
{
  if (__first != __last)
  {
    typename iterator_traits<_InputIterator>::value_type __t(*__first);
    *__result = __t;
    for (++__first, (void) ++__result; __first != __last; ++__first, (void) ++__result)
    {
      __t       = ::cuda::std::move(__t) + *__first;
      *__result = __t;
    }
  }
  return __result;
}

template <class _InputIterator, class _OutputIterator, class _BinaryOperation>
_CCCL_API constexpr _OutputIterator
partial_sum(_InputIterator __first, _InputIterator __last, _OutputIterator __result, _BinaryOperation __binary_op)
{
  if (__first != __last)
  {
    typename iterator_traits<_InputIterator>::value_type __t(*__first);
    *__result = __t;
    for (++__first, (void) ++__result; __first != __last; ++__first, (void) ++__result)
    {
      __t       = __binary_op(::cuda::std::move(__t), *__first);
      *__result = __t;
    }
  }
  return __result;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___NUMERIC_PARTIAL_SUM_H
