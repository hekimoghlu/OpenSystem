/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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

#ifndef _CUDA_STD___FUNCTIONAL_PERFECT_FORWARD_H
#define _CUDA_STD___FUNCTIONAL_PERFECT_FORWARD_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__concepts/concept_macros.h>
#include <uscl/std/__functional/invoke.h>
#include <uscl/std/__type_traits/enable_if.h>
#include <uscl/std/__type_traits/is_constructible.h>
#include <uscl/std/__type_traits/is_nothrow_constructible.h>
#include <uscl/std/__utility/declval.h>
#include <uscl/std/__utility/forward.h>
#include <uscl/std/__utility/integer_sequence.h>
#include <uscl/std/__utility/move.h>
#include <uscl/std/tuple>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Op, class _Indices, class... _BoundArgs>
struct __perfect_forward_impl;

template <class _Op, size_t... _Idx, class... _BoundArgs>
struct __perfect_forward_impl<_Op, index_sequence<_Idx...>, _BoundArgs...>
{
private:
  tuple<_BoundArgs...> __bound_args_;

  template <class... _Args>
  static constexpr bool __noexcept_constructible = is_nothrow_constructible_v<tuple<_BoundArgs...>, _Args&&...>;

public:
  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(is_constructible_v<tuple<_BoundArgs...>, _Args&&...>)
  _CCCL_API explicit constexpr __perfect_forward_impl(_Args&&... __bound_args) noexcept(
    __noexcept_constructible<_Args...>)
      : __bound_args_(::cuda::std::forward<_Args>(__bound_args)...)
  {}

  _CCCL_HIDE_FROM_ABI __perfect_forward_impl(__perfect_forward_impl const&) = default;
  _CCCL_HIDE_FROM_ABI __perfect_forward_impl(__perfect_forward_impl&&)      = default;

  _CCCL_HIDE_FROM_ABI __perfect_forward_impl& operator=(__perfect_forward_impl const&) = default;
  _CCCL_HIDE_FROM_ABI __perfect_forward_impl& operator=(__perfect_forward_impl&&)      = default;

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(is_invocable_v<_Op, _BoundArgs&..., _Args...>)
  _CCCL_API constexpr auto operator()(_Args&&... __args) & noexcept(
    noexcept(_Op()(::cuda::std::get<_Idx>(__bound_args_)..., ::cuda::std::forward<_Args>(__args)...)))
    -> decltype(_Op()(::cuda::std::get<_Idx>(__bound_args_)..., ::cuda::std::forward<_Args>(__args)...))
  {
    return _Op()(::cuda::std::get<_Idx>(__bound_args_)..., ::cuda::std::forward<_Args>(__args)...);
  }

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES((!is_invocable_v<_Op, _BoundArgs&..., _Args...>) )
  _CCCL_API inline auto operator()(_Args&&...) & = delete;

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(is_invocable_v<_Op, _BoundArgs const&..., _Args...>)
  _CCCL_API constexpr auto operator()(_Args&&... __args) const& noexcept(
    noexcept(_Op()(::cuda::std::get<_Idx>(__bound_args_)..., ::cuda::std::forward<_Args>(__args)...)))
    -> decltype(_Op()(::cuda::std::get<_Idx>(__bound_args_)..., ::cuda::std::forward<_Args>(__args)...))
  {
    return _Op()(::cuda::std::get<_Idx>(__bound_args_)..., ::cuda::std::forward<_Args>(__args)...);
  }

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES((!is_invocable_v<_Op, _BoundArgs const&..., _Args...>) )
  _CCCL_API inline auto operator()(_Args&&...) const& = delete;

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(is_invocable_v<_Op, _BoundArgs..., _Args...>)
  _CCCL_API constexpr auto operator()(_Args&&... __args) && noexcept(noexcept(
    _Op()(::cuda::std::get<_Idx>(::cuda::std::move(__bound_args_))..., ::cuda::std::forward<_Args>(__args)...)))
    -> decltype(_Op()(::cuda::std::get<_Idx>(::cuda::std::move(__bound_args_))...,
                      ::cuda::std::forward<_Args>(__args)...))
  {
    return _Op()(::cuda::std::get<_Idx>(::cuda::std::move(__bound_args_))..., ::cuda::std::forward<_Args>(__args)...);
  }

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES((!is_invocable_v<_Op, _BoundArgs..., _Args...>) )
  _CCCL_API inline auto operator()(_Args&&...) && = delete;

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(is_invocable_v<_Op, _BoundArgs const..., _Args...>)
  _CCCL_API constexpr auto operator()(_Args&&... __args) const&& noexcept(noexcept(
    _Op()(::cuda::std::get<_Idx>(::cuda::std::move(__bound_args_))..., ::cuda::std::forward<_Args>(__args)...)))
    -> decltype(_Op()(::cuda::std::get<_Idx>(::cuda::std::move(__bound_args_))...,
                      ::cuda::std::forward<_Args>(__args)...))
  {
    return _Op()(::cuda::std::get<_Idx>(::cuda::std::move(__bound_args_))..., ::cuda::std::forward<_Args>(__args)...);
  }

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES((!is_invocable_v<_Op, _BoundArgs const..., _Args...>) )
  _CCCL_API inline auto operator()(_Args&&...) const&& = delete;
};

// __perfect_forward implements a perfect-forwarding call wrapper as explained in [func.require].
template <class _Op, class... _Args>
using __perfect_forward = __perfect_forward_impl<_Op, index_sequence_for<_Args...>, _Args...>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FUNCTIONAL_PERFECT_FORWARD_H
