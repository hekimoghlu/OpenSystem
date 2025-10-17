/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_STATIC_FOR_H
#define _CUDA___UTILITY_STATIC_FOR_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__type_traits/integral_constant.h>
#include <uscl/std/__utility/forward.h>
#include <uscl/std/__utility/integer_sequence.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _SizeType, _SizeType _Start, _SizeType _Step, typename _Operator, _SizeType... _Indices, typename... _TArgs>
_CCCL_API constexpr void
__static_for_impl(_Operator __op, ::cuda::std::integer_sequence<_SizeType, _Indices...>, _TArgs&&... __args) noexcept(
  (true && ... && noexcept(__op(::cuda::std::integral_constant<_SizeType, (_Indices * _Step + _Start)>{}, __args...))))
{
  (__op(::cuda::std::integral_constant<_SizeType, (_Indices * _Step + _Start)>{}, __args...), ...);
}

template <typename _Tp, _Tp _Size, typename _Operator, typename... _TArgs>
_CCCL_API constexpr void
static_for(_Operator __op, _TArgs&&... __args) noexcept(noexcept(::cuda::__static_for_impl<_Tp, 0, 1>(
  __op, ::cuda::std::make_integer_sequence<_Tp, _Size>{}, ::cuda::std::forward<_TArgs>(__args)...)))
{
  ::cuda::__static_for_impl<_Tp, 0, 1>(
    __op, ::cuda::std::make_integer_sequence<_Tp, _Size>{}, ::cuda::std::forward<_TArgs>(__args)...);
}

template <typename _Tp, _Tp _Start, _Tp _End, _Tp _Step = 1, typename _Operator, typename... _TArgs>
_CCCL_API constexpr void
static_for(_Operator __op, _TArgs&&... __args) noexcept(noexcept(::cuda::__static_for_impl<_Tp, _Start, _Step>(
  __op, ::cuda::std::make_integer_sequence<_Tp, (_End - _Start) / _Step>{}, ::cuda::std::forward<_TArgs>(__args)...)))
{
  ::cuda::__static_for_impl<_Tp, _Start, _Step>(
    __op, ::cuda::std::make_integer_sequence<_Tp, (_End - _Start) / _Step>{}, ::cuda::std::forward<_TArgs>(__args)...);
}

template <auto _Size, typename _Operator, typename... _TArgs>
_CCCL_API constexpr void static_for(_Operator __op, _TArgs&&... __args) noexcept(
  noexcept(::cuda::static_for<decltype(_Size), _Size>(__op, ::cuda::std::forward<_TArgs>(__args)...)))
{
  ::cuda::static_for<decltype(_Size), _Size>(__op, ::cuda::std::forward<_TArgs>(__args)...);
}

template <auto _Start,
          decltype(_Start) _End,
          decltype(_Start) _Step = decltype(_Start){1},
          typename _Operator,
          typename... _TArgs>
_CCCL_API constexpr void static_for(_Operator __op, _TArgs&&... __args) noexcept(
  noexcept(::cuda::static_for<decltype(_Start), _Start, _End, _Step>(__op, ::cuda::std::forward<_TArgs>(__args)...)))
{
  ::cuda::static_for<decltype(_Start), _Start, _End, _Step>(__op, ::cuda::std::forward<_TArgs>(__args)...);
}

_CCCL_END_NAMESPACE_CUDA

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_STATIC_FOR_H
