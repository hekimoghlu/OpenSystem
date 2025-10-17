/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 18, 2024.
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

#ifndef _CUDA_STD___ALGORITHM_UNIQUE_COPY_H
#define _CUDA_STD___ALGORITHM_UNIQUE_COPY_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__algorithm/comp.h>
#include <uscl/std/__algorithm/iterator_operations.h>
#include <uscl/std/__iterator/iterator_traits.h>
#include <uscl/std/__type_traits/conditional.h>
#include <uscl/std/__type_traits/is_base_of.h>
#include <uscl/std/__type_traits/is_same.h>
#include <uscl/std/__utility/move.h>
#include <uscl/std/__utility/pair.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace __unique_copy_tags
{

struct __reread_from_input_tag
{};
struct __reread_from_output_tag
{};
struct __read_from_tmp_value_tag
{};

} // namespace __unique_copy_tags

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _BinaryPredicate, class _InputIterator, class _Sent, class _OutputIterator>
constexpr _CCCL_API inline pair<_InputIterator, _OutputIterator> __unique_copy(
  _InputIterator __first,
  _Sent __last,
  _OutputIterator __result,
  _BinaryPredicate&& __pred,
  __unique_copy_tags::__read_from_tmp_value_tag)
{
  if (__first != __last)
  {
    typename _IterOps<_AlgPolicy>::template __value_type<_InputIterator> __t(*__first);
    *__result = __t;
    ++__result;
    while (++__first != __last)
    {
      if (!__pred(__t, *__first))
      {
        __t       = *__first;
        *__result = __t;
        ++__result;
      }
    }
  }
  return pair<_InputIterator, _OutputIterator>(::cuda::std::move(__first), ::cuda::std::move(__result));
}

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _BinaryPredicate, class _ForwardIterator, class _Sent, class _OutputIterator>
constexpr _CCCL_API inline pair<_ForwardIterator, _OutputIterator> __unique_copy(
  _ForwardIterator __first,
  _Sent __last,
  _OutputIterator __result,
  _BinaryPredicate&& __pred,
  __unique_copy_tags::__reread_from_input_tag)
{
  if (__first != __last)
  {
    _ForwardIterator __i = __first;
    *__result            = *__i;
    ++__result;
    while (++__first != __last)
    {
      if (!__pred(*__i, *__first))
      {
        *__result = *__first;
        ++__result;
        __i = __first;
      }
    }
  }
  return pair<_ForwardIterator, _OutputIterator>(::cuda::std::move(__first), ::cuda::std::move(__result));
}

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _BinaryPredicate, class _InputIterator, class _Sent, class _InputAndOutputIterator>
constexpr _CCCL_API inline pair<_InputIterator, _InputAndOutputIterator> __unique_copy(
  _InputIterator __first,
  _Sent __last,
  _InputAndOutputIterator __result,
  _BinaryPredicate&& __pred,
  __unique_copy_tags::__reread_from_output_tag)
{
  if (__first != __last)
  {
    *__result = *__first;
    while (++__first != __last)
    {
      if (!__pred(*__result, *__first))
      {
        *++__result = *__first;
      }
    }
    ++__result;
  }
  return pair<_InputIterator, _InputAndOutputIterator>(::cuda::std::move(__first), ::cuda::std::move(__result));
}

_CCCL_EXEC_CHECK_DISABLE
template <class _InputIterator, class _OutputIterator, class _BinaryPredicate>
_CCCL_API constexpr _OutputIterator
unique_copy(_InputIterator __first, _InputIterator __last, _OutputIterator __result, _BinaryPredicate __pred)
{
  using __algo_tag =
    conditional_t<is_base_of_v<forward_iterator_tag, __iterator_category_type<_InputIterator>>,
                  __unique_copy_tags::__reread_from_input_tag,
                  conditional_t<is_base_of_v<forward_iterator_tag, __iterator_category_type<_OutputIterator>>
                                  && is_same_v<__iter_value_type<_InputIterator>, __iter_value_type<_OutputIterator>>,
                                __unique_copy_tags::__reread_from_output_tag,
                                __unique_copy_tags::__read_from_tmp_value_tag>>;
  return ::cuda::std::__unique_copy<_ClassicAlgPolicy>(
           ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__result), __pred, __algo_tag())
    .second;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _InputIterator, class _OutputIterator>
_CCCL_API constexpr _OutputIterator unique_copy(_InputIterator __first, _InputIterator __last, _OutputIterator __result)
{
  return ::cuda::std::unique_copy(
    ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__result), __equal_to{});
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_UNIQUE_COPY_H
