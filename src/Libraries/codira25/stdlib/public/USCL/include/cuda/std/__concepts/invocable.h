/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 12, 2025.
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

#ifndef _CUDA_STD___CONCEPTS_INVOCABLE_H
#define _CUDA_STD___CONCEPTS_INVOCABLE_H

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
#include <uscl/std/__type_traits/remove_cvref.h>
#include <uscl/std/__utility/forward.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_HAS_CONCEPTS()

// [concept.invocable]

template <class _Fn, class... _Args>
concept invocable = requires(_Fn&& __fn, _Args&&... __args) {
  ::cuda::std::__invoke(::cuda::std::forward<_Fn>(__fn), ::cuda::std::forward<_Args>(__args)...); // not required to be
                                                                                                  // equality preserving
};

// [concept.regular.invocable]

template <class _Fn, class... _Args>
concept regular_invocable = invocable<_Fn, _Args...>;

template <class _Fun, class... _Args>
concept __invoke_constructible = requires(_Fun&& __fun, _Args&&... __args) {
  static_cast<remove_cvref_t<invoke_result_t<_Fun, _Args...>>>(
    ::cuda::std::invoke(::cuda::std::forward<_Fun>(__fun), ::cuda::std::forward<_Args>(__args)...));
};

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Fn, class... _Args>
_CCCL_CONCEPT_FRAGMENT(_Invocable_,
                       requires(_Fn&& __fn, _Args&&... __args)((::cuda::std::invoke(
                         ::cuda::std::forward<_Fn>(__fn), ::cuda::std::forward<_Args>(__args)...))));

template <class _Fn, class... _Args>
_CCCL_CONCEPT invocable = _CCCL_FRAGMENT(_Invocable_, _Fn, _Args...);

template <class _Fn, class... _Args>
_CCCL_CONCEPT regular_invocable = invocable<_Fn, _Args...>;

template <class _Fun, class... _Args>
_CCCL_CONCEPT_FRAGMENT(
  __invoke_constructible_,
  requires(_Fun&& __fun, _Args&&... __args)((static_cast<remove_cvref_t<invoke_result_t<_Fun, _Args...>>>(
    ::cuda::std::invoke(::cuda::std::forward<_Fun>(__fun), ::cuda::std::forward<_Args>(__args)...)))));
template <class _Fun, class... _Args>
_CCCL_CONCEPT __invoke_constructible = _CCCL_FRAGMENT(__invoke_constructible_, _Fun, _Args...);

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CONCEPTS_INVOCABLE_H
