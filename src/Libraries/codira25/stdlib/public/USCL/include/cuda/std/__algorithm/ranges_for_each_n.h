/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 16, 2021.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_RANGES_FOR_EACH_N_H
#define _CUDA_STD___ALGORITHM_RANGES_FOR_EACH_N_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__algorithm/in_fun_result.h>
#include <uscl/std/__functional/identity.h>
#include <uscl/std/__functional/invoke.h>
#include <uscl/std/__iterator/concepts.h>
#include <uscl/std/__iterator/incrementable_traits.h>
#include <uscl/std/__iterator/iterator_traits.h>
#include <uscl/std/__iterator/projected.h>
#include <uscl/std/__ranges/concepts.h>
#include <uscl/std/__utility/move.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_RANGES

template <class _Iter, class _Func>
using for_each_n_result = in_fun_result<_Iter, _Func>;

_CCCL_BEGIN_NAMESPACE_CPO(__for_each_n)

struct __fn
{
  _CCCL_TEMPLATE(class _Iter, class _Func, class _Proj = identity)
  _CCCL_REQUIRES(input_iterator<_Iter> _CCCL_AND indirectly_unary_invocable<_Func, projected<_Iter, _Proj>>)
  _CCCL_API constexpr for_each_n_result<_Iter, _Func>
  operator()(_Iter __first, iter_difference_t<_Iter> __count, _Func __func, _Proj __proj = {}) const
  {
    while (__count-- > 0)
    {
      ::cuda::std::invoke(__func, ::cuda::std::invoke(__proj, *__first));
      ++__first;
    }
    return {::cuda::std::move(__first), ::cuda::std::move(__func)};
  }
};
_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto for_each_n = __for_each_n::__fn{};
} // namespace __cpo

_CCCL_END_NAMESPACE_RANGES

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_RANGES_FOR_EACH_N_H
