/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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

// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-25 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FUNCTIONAL_BIND_BACK_H
#define _CUDA_STD___FUNCTIONAL_BIND_BACK_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__functional/invoke.h>
#include <uscl/std/__functional/perfect_forward.h>
#include <uscl/std/__fwd/get.h>
#include <uscl/std/__tuple_dir/tuple_size.h>
#include <uscl/std/__type_traits/decay.h>
#include <uscl/std/__type_traits/enable_if.h>
#include <uscl/std/__type_traits/is_constructible.h>
#include <uscl/std/__type_traits/is_move_constructible.h>
#include <uscl/std/__utility/forward.h>
#include <uscl/std/__utility/integer_sequence.h>
#include <uscl/std/tuple>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <size_t _NBound, class = make_index_sequence<_NBound>>
struct __bind_back_op;

template <size_t _NBound, size_t... _Ip>
struct __bind_back_op<_NBound, index_sequence<_Ip...>>
{
  // clang-format off
  template <class _Fn, class _BoundArgs, class... _Args>
  _CCCL_API constexpr auto
  operator()(_Fn&& __f, _BoundArgs&& __bound_args, _Args&&... __args) const
  noexcept(noexcept(::cuda::std::invoke(::cuda::std::forward<_Fn>(__f), ::cuda::std::forward<_Args>(__args)..., ::cuda::std::get<_Ip>(::cuda::std::forward<_BoundArgs>(__bound_args))...)))
  -> decltype(      ::cuda::std::invoke(::cuda::std::forward<_Fn>(__f), ::cuda::std::forward<_Args>(__args)..., ::cuda::std::get<_Ip>(::cuda::std::forward<_BoundArgs>(__bound_args))...))
  { return          ::cuda::std::invoke(::cuda::std::forward<_Fn>(__f), ::cuda::std::forward<_Args>(__args)..., ::cuda::std::get<_Ip>(::cuda::std::forward<_BoundArgs>(__bound_args))...); }
  // clang-format on
};

template <class _Fn, class _BoundArgs>
struct __bind_back_t : __perfect_forward<__bind_back_op<tuple_size_v<_BoundArgs>>, _Fn, _BoundArgs>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(
    __bind_back_t, __perfect_forward, __bind_back_op<tuple_size_v<_BoundArgs>>, _Fn, _BoundArgs);
};

template <class _Fn,
          class... _Args,
          class = enable_if_t<_And<is_constructible<decay_t<_Fn>, _Fn>,
                                   is_move_constructible<decay_t<_Fn>>,
                                   is_constructible<decay_t<_Args>, _Args>...,
                                   is_move_constructible<decay_t<_Args>>...>::value>>
// clang-format off
_CCCL_API constexpr auto __bind_back(_Fn&& __f, _Args&&... __args)
    noexcept(noexcept(__bind_back_t<decay_t<_Fn>, tuple<decay_t<_Args>...>>(::cuda::std::forward<_Fn>(__f), ::cuda::std::forward_as_tuple(::cuda::std::forward<_Args>(__args)...))))
    -> decltype(      __bind_back_t<decay_t<_Fn>, tuple<decay_t<_Args>...>>(::cuda::std::forward<_Fn>(__f), ::cuda::std::forward_as_tuple(::cuda::std::forward<_Args>(__args)...)))
    { return          __bind_back_t<decay_t<_Fn>, tuple<decay_t<_Args>...>>(::cuda::std::forward<_Fn>(__f), ::cuda::std::forward_as_tuple(::cuda::std::forward<_Args>(__args)...)); }
// clang-format on

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FUNCTIONAL_BIND_BACK_H
