/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 25, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_COPY_H
#define _CUDA_STD___ALGORITHM_COPY_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__algorithm/iterator_operations.h>
#include <uscl/std/__algorithm/unwrap_iter.h>
#include <uscl/std/__type_traits/enable_if.h>
#include <uscl/std/__type_traits/is_constant_evaluated.h>
#include <uscl/std/__type_traits/is_same.h>
#include <uscl/std/__type_traits/is_trivially_copyable.h>
#include <uscl/std/__type_traits/remove_const.h>
#include <uscl/std/cstdint>
#include <uscl/std/cstdlib>
#include <uscl/std/cstring> // memmove

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _InputIterator, class _OutputIterator>
_CCCL_API constexpr pair<_InputIterator, _OutputIterator>
__copy(_InputIterator __first, _InputIterator __last, _OutputIterator __result)
{
  for (; __first != __last; ++__first, (void) ++__result)
  {
    *__result = *__first;
  }
  return {__last, __result};
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr bool __dispatch_memmove(_Up* __result, _Tp* __first, const size_t __n)
{
#if defined(_CCCL_BUILTIN_MEMMOVE)
  _CCCL_BUILTIN_MEMMOVE(__result, __first, __n * sizeof(_Up));
  return true;
#else // ^^^ _CCCL_BUILTIN_MEMMOVE ^^^ / vvv !_CCCL_BUILTIN_MEMMOVE vvv
  if (!::cuda::std::__cccl_default_is_constant_evaluated())
  {
    ::cuda::std::memmove(__result, __first, __n * sizeof(_Up));
    return true;
  }

  return false;
#endif // ^^^ !_CCCL_BUILTIN_MEMMOVE ^^^
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr bool __constexpr_tail_overlap_fallback(_Tp* __first, _Up* __needle, _Tp* __last)
{
  while (__first != __last)
  {
    if (__first == __needle)
    {
      return true;
    }
    ++__first;
  }
  return false;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr bool __constexpr_tail_overlap(_Tp* __first, _Up* __needle, [[maybe_unused]] _Tp* __last)
{
#if defined(_CCCL_BUILTIN_CONSTANT_P)
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return _CCCL_BUILTIN_CONSTANT_P(__first < __needle) && __first < __needle;),
                    (return __constexpr_tail_overlap_fallback(__first, __needle, __last);))
#else // ^^^ _CCCL_BUILTIN_CONSTANT_P ^^^ / vvv !_CCCL_BUILTIN_CONSTANT_P vvv
  return __constexpr_tail_overlap_fallback(__first, __needle, __last);
#endif // !_CCCL_BUILTIN_CONSTANT_P
}

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy,
          class _Tp,
          class _Up,
          enable_if_t<is_same_v<remove_const_t<_Tp>, _Up>, int> = 0,
          enable_if_t<is_trivially_copyable_v<_Up>, int>        = 0>
_CCCL_API constexpr pair<_Tp*, _Up*> __copy(_Tp* __first, _Tp* __last, _Up* __result)
{
  const ptrdiff_t __n = __last - __first;
  if (__n > 0)
  {
    if (__dispatch_memmove(__result, __first, __n))
    {
      return {__last, __result + __n};
    }
    if ((!::cuda::std::is_constant_evaluated() && __first < __result)
        || __constexpr_tail_overlap(__first, __result, __last))
    {
      for (ptrdiff_t __i = __n; __i > 0; --__i)
      {
        *(__result + __i - 1) = *(__first + __i - 1);
      }
    }
    else
    {
      for (ptrdiff_t __i = 0; __i < __n; ++__i)
      {
        *(__result + __i) = *(__first + __i);
      }
    }
  }
  return {__last, __result + __n};
}

template <class _InputIterator, class _OutputIterator>
_CCCL_API constexpr _OutputIterator copy(_InputIterator __first, _InputIterator __last, _OutputIterator __result)
{
  return ::cuda::std::__copy<_ClassicAlgPolicy>(
           ::cuda::std::__unwrap_iter(__first), ::cuda::std::__unwrap_iter(__last), ::cuda::std::__unwrap_iter(__result))
    .second;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_COPY_H
