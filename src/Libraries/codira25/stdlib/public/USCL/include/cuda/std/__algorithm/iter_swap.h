/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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

#ifndef _CUDA_STD___ALGORITHM_ITER_SWAP_H
#define _CUDA_STD___ALGORITHM_ITER_SWAP_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__concepts/concept_macros.h>
#include <uscl/std/__utility/declval.h>
#include <uscl/std/__utility/forward.h>
#include <uscl/std/__utility/swap.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

//! Intentionally not an algorithm to avoid breaking types that pull in `::std::iter_swap` via ADL
_CCCL_BEGIN_NAMESPACE_CPO(__iter_swap)
// "Poison pill" overload to intentionally create ambiguity with the unconstrained
// `std::iter_swap` function.
template <class _ForwardIterator1, class _ForwardIterator2>
void iter_swap(_ForwardIterator1, _ForwardIterator2) = delete;

template <class _ForwardIterator1, class _ForwardIterator2>
_CCCL_CONCEPT __unqualified_iter_swap =
  _CCCL_REQUIRES_EXPR((_ForwardIterator1, _ForwardIterator2), _ForwardIterator1&& __a, _ForwardIterator2&& __b)(
    iter_swap(::cuda::std::forward<_ForwardIterator1>(__a), ::cuda::std::forward<_ForwardIterator2>(__b)));

template <class _ForwardIterator1, class _ForwardIterator2>
_CCCL_CONCEPT __readable_swappable =
  _CCCL_REQUIRES_EXPR((_ForwardIterator1, _ForwardIterator2), _ForwardIterator1 __a, _ForwardIterator2 __b)(
    requires(!__unqualified_iter_swap<_ForwardIterator1, _ForwardIterator2>), swap(*__a, *__b));

struct __fn
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _ForwardIterator1, class _ForwardIterator2)
  _CCCL_REQUIRES(__unqualified_iter_swap<_ForwardIterator1, _ForwardIterator2>)
  _CCCL_API constexpr void operator()(_ForwardIterator1&& __a, _ForwardIterator2&& __b) const
    noexcept(noexcept(iter_swap(::cuda::std::declval<_ForwardIterator1>(), ::cuda::std::declval<_ForwardIterator2>())))
  {
    (void) iter_swap(::cuda::std::forward<_ForwardIterator1>(__a), ::cuda::std::forward<_ForwardIterator2>(__b));
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _ForwardIterator1, class _ForwardIterator2)
  _CCCL_REQUIRES(__readable_swappable<_ForwardIterator1, _ForwardIterator2>)
  _CCCL_API constexpr void operator()(_ForwardIterator1&& __a, _ForwardIterator2&& __b) const
    noexcept(noexcept(swap(*::cuda::std::declval<_ForwardIterator1>(), *::cuda::std::declval<_ForwardIterator2>())))
  {
    swap(*__a, *__b);
  }
};

_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
// This is a global constant to avoid breaking types that pull in `::std::iter_swap` via ADL
_CCCL_GLOBAL_CONSTANT auto iter_swap = __iter_swap::__fn{};
} // namespace __cpo

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_ITER_SWAP_H
