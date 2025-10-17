/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 10, 2022.
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

#ifndef _CUDA_STD___CONCEPTS_DESTRUCTIBLE_H
#define _CUDA_STD___CONCEPTS_DESTRUCTIBLE_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__concepts/concept_macros.h>
#include <uscl/std/__type_traits/enable_if.h>
#include <uscl/std/__type_traits/is_destructible.h>
#include <uscl/std/__type_traits/is_nothrow_destructible.h>
#include <uscl/std/__type_traits/is_object.h>
#include <uscl/std/__type_traits/void_t.h>
#include <uscl/std/__utility/declval.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_COMPILER(MSVC)

template <class _Tp>
_CCCL_CONCEPT destructible = __is_nothrow_destructible(_Tp);

#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv

template <class _Tp, class = void, class = void>
inline constexpr bool __destructible_impl = false;

template <class _Tp>
inline constexpr bool __destructible_impl<_Tp,
                                          enable_if_t<is_object_v<_Tp>>,
#  if _CCCL_COMPILER(GCC)
                                          enable_if_t<is_destructible_v<_Tp>>>
#  else // ^^^ _CCCL_COMPILER(GCC) ^^^ / vvv !_CCCL_COMPILER(GCC) vvv
                                          void_t<decltype(::cuda::std::declval<_Tp>().~_Tp())>>
#  endif // !_CCCL_COMPILER(GCC)
  = noexcept(::cuda::std::declval<_Tp>().~_Tp());

template <class _Tp>
inline constexpr bool __destructible = __destructible_impl<_Tp>;

template <class _Tp>
inline constexpr bool __destructible<_Tp&> = true;

template <class _Tp>
inline constexpr bool __destructible<_Tp&&> = true;

template <class _Tp, size_t _Nm>
inline constexpr bool __destructible<_Tp[_Nm]> = __destructible<_Tp>;

template <class _Tp>
_CCCL_CONCEPT destructible = __destructible<_Tp>;

#endif // !_CCCL_COMPILER(MSVC)

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CONCEPTS_DESTRUCTIBLE_H
