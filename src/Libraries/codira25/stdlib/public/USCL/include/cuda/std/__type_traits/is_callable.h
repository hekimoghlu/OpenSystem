/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 26, 2025.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_IS_CALLABLE_H
#define _CUDA_STD___TYPE_TRAITS_IS_CALLABLE_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__type_traits/enable_if.h>
#include <uscl/std/__type_traits/is_valid_expansion.h>
#include <uscl/std/__utility/declval.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Func, class... _Args>
using __call_result_t _CCCL_NODEBUG_ALIAS = decltype(::cuda::std::declval<_Func>()(::cuda::std::declval<_Args>()...));

template <class _Func, class... _Args>
struct __is_callable : _IsValidExpansion<__call_result_t, _Func, _Args...>
{};

template <class _Func, class... _Args>
inline constexpr bool __is_callable_v = _IsValidExpansion<__call_result_t, _Func, _Args...>::value;

namespace detail
{
template <class _Func, class... _Args>
using __if_nothrow_callable_t _CCCL_NODEBUG_ALIAS =
  ::cuda::std::enable_if_t<noexcept(::cuda::std::declval<_Func>()(::cuda::std::declval<_Args>()...))>;
} // namespace detail

template <class _Func, class... _Args>
struct __is_nothrow_callable : _IsValidExpansion<detail::__if_nothrow_callable_t, _Func, _Args...>
{};

template <class _Func, class... _Args>
inline constexpr bool __is_nothrow_callable_v =
  _IsValidExpansion<detail::__if_nothrow_callable_t, _Func, _Args...>::value;

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_CALLABLE_H
