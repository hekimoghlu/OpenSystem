/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 28, 2025.
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

#ifndef _CUDA_STD___ALGORITHM_MAKE_PROJECTED_H
#define _CUDA_STD___ALGORITHM_MAKE_PROJECTED_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__concepts/same_as.h>
#include <uscl/std/__functional/identity.h>
#include <uscl/std/__functional/invoke.h>
#include <uscl/std/__type_traits/decay.h>
#include <uscl/std/__type_traits/enable_if.h>
#include <uscl/std/__type_traits/integral_constant.h>
#include <uscl/std/__type_traits/is_member_pointer.h>
#include <uscl/std/__type_traits/is_same.h>
#include <uscl/std/__utility/declval.h>
#include <uscl/std/__utility/forward.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Pred, class _Proj>
struct _ProjectedPred
{
  _Pred& __pred; // Can be a unary or a binary predicate.
  _Proj& __proj;

  _CCCL_API constexpr _ProjectedPred(_Pred& __pred_arg, _Proj& __proj_arg)
      : __pred(__pred_arg)
      , __proj(__proj_arg)
  {}

  template <class _Tp>
  typename __invoke_of<
    _Pred&,
    decltype(::cuda::std::__invoke(::cuda::std::declval<_Proj&>(), ::cuda::std::declval<_Tp>()))>::type constexpr
    _CCCL_API inline
    operator()(_Tp&& __v) const
  {
    return ::cuda::std::__invoke(__pred, ::cuda::std::__invoke(__proj, ::cuda::std::forward<_Tp>(__v)));
  }

  template <class _T1, class _T2>
  typename __invoke_of<
    _Pred&,
    decltype(::cuda::std::__invoke(::cuda::std::declval<_Proj&>(), ::cuda::std::declval<_T1>())),
    decltype(::cuda::std::__invoke(::cuda::std::declval<_Proj&>(), ::cuda::std::declval<_T2>()))>::type constexpr
    _CCCL_API inline
    operator()(_T1&& __lhs, _T2&& __rhs) const
  {
    return ::cuda::std::__invoke(__pred,
                                 ::cuda::std::__invoke(__proj, ::cuda::std::forward<_T1>(__lhs)),
                                 ::cuda::std::__invoke(__proj, ::cuda::std::forward<_T2>(__rhs)));
  }
};

template <class _Pred,
          class _Proj,
          enable_if_t<!(!is_member_pointer<decay_t<_Pred>>::value && __is_identity<decay_t<_Proj>>::value), int> = 0>
_CCCL_API constexpr _ProjectedPred<_Pred, _Proj> __make_projected(_Pred& __pred, _Proj& __proj)
{
  return _ProjectedPred<_Pred, _Proj>(__pred, __proj);
}

// Avoid creating the functor and just use the pristine comparator -- for certain algorithms, this would enable
// optimizations that rely on the type of the comparator. Additionally, this results in less layers of indirection in
// the call stack when the comparator is invoked, even in an unoptimized build.
template <class _Pred,
          class _Proj,
          enable_if_t<!is_member_pointer<decay_t<_Pred>>::value && __is_identity<decay_t<_Proj>>::value, int> = 0>
_CCCL_API constexpr _Pred& __make_projected(_Pred& __pred, _Proj&)
{
  return __pred;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_MAKE_PROJECTED_H
