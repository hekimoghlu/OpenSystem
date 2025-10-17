/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA_STD___RANGES_ALL_H
#define _CUDA_STD___RANGES_ALL_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__iterator/concepts.h>
#include <uscl/std/__iterator/iterator_traits.h>
#include <uscl/std/__ranges/access.h>
#include <uscl/std/__ranges/concepts.h>
#include <uscl/std/__ranges/owning_view.h>
#include <uscl/std/__ranges/range_adaptor.h>
#include <uscl/std/__ranges/ref_view.h>
#include <uscl/std/__type_traits/decay.h>
#include <uscl/std/__utility/auto_cast.h>
#include <uscl/std/__utility/declval.h>
#include <uscl/std/__utility/forward.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_VIEWS

_CCCL_BEGIN_NAMESPACE_CPO(__all)

template <class _Tp>
_CCCL_CONCEPT __to_ref_view = _CCCL_REQUIRES_EXPR((_Tp), _Tp&& __t)(
  requires(!::cuda::std::ranges::view<decay_t<_Tp>>), (::cuda::std::ranges::ref_view{::cuda::std::forward<_Tp>(__t)}));

template <class _Tp>
_CCCL_CONCEPT __to_owning_view = _CCCL_REQUIRES_EXPR((_Tp), _Tp&& __t)(
  requires(!::cuda::std::ranges::view<decay_t<_Tp>>),
  requires(!__to_ref_view<_Tp>),
  (::cuda::std::ranges::owning_view{::cuda::std::forward<_Tp>(__t)}));

struct __fn : __range_adaptor_closure<__fn>
{
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::ranges::view<decay_t<_Tp>>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __t) const
    noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(::cuda::std::forward<_Tp>(__t))))
      -> decltype(_LIBCUDACXX_AUTO_CAST(::cuda::std::forward<_Tp>(__t)))
  {
    return _LIBCUDACXX_AUTO_CAST(::cuda::std::forward<_Tp>(__t));
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__to_ref_view<_Tp>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __t) const
    noexcept(noexcept(::cuda::std::ranges::ref_view{::cuda::std::forward<_Tp>(__t)}))
  {
    return ::cuda::std::ranges::ref_view{::cuda::std::forward<_Tp>(__t)};
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__to_owning_view<_Tp>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __t) const
    noexcept(noexcept(::cuda::std::ranges::owning_view{::cuda::std::forward<_Tp>(__t)}))
  {
    return ::cuda::std::ranges::owning_view{::cuda::std::forward<_Tp>(__t)};
  }
};
_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto all = __all::__fn{};
} // namespace __cpo

#if _CCCL_HAS_CONCEPTS()
template <::cuda::std::ranges::viewable_range _Range>
using all_t = decltype(::cuda::std::ranges::views::all(declval<_Range>()));
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Range>
using all_t =
  enable_if_t<::cuda::std::ranges::viewable_range<_Range>, decltype(::cuda::std::ranges::views::all(declval<_Range>()))>;
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_CCCL_END_NAMESPACE_VIEWS

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANGES_ALL_H
