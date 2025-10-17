/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 30, 2023.
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

#ifndef _CUDA_STD___CONCEPTS_COMMON_WITH_H
#define _CUDA_STD___CONCEPTS_COMMON_WITH_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__concepts/common_reference_with.h>
#include <uscl/std/__concepts/concept_macros.h>
#include <uscl/std/__concepts/same_as.h>
#include <uscl/std/__type_traits/add_lvalue_reference.h>
#include <uscl/std/__type_traits/common_type.h>
#include <uscl/std/__utility/declval.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_HAS_CONCEPTS()

// [concept.common]

template <class _Tp, class _Up>
concept common_with = same_as<common_type_t<_Tp, _Up>, common_type_t<_Up, _Tp>> && requires {
  static_cast<common_type_t<_Tp, _Up>>(::cuda::std::declval<_Tp>());
  static_cast<common_type_t<_Tp, _Up>>(::cuda::std::declval<_Up>());
} && common_reference_with<add_lvalue_reference_t<const _Tp>, add_lvalue_reference_t<const _Up>> && common_reference_with<add_lvalue_reference_t<common_type_t<_Tp, _Up>>, common_reference_t<add_lvalue_reference_t<const _Tp>, add_lvalue_reference_t<const _Up>>>;

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Tp, class _Up>
_CCCL_CONCEPT_FRAGMENT(__common_type_exists_,
                       requires()(typename(common_type_t<_Tp, _Up>), typename(common_type_t<_Up, _Tp>)));

template <class _Tp, class _Up>
_CCCL_CONCEPT _Common_type_exists = _CCCL_FRAGMENT(__common_type_exists_, _Tp, _Up);

template <class _Tp, class _Up>
_CCCL_CONCEPT_FRAGMENT(__common_type_constructible_,
                       requires()(requires(_Common_type_exists<_Tp, _Up>),
                                  (static_cast<common_type_t<_Tp, _Up>>(::cuda::std::declval<_Tp>())),
                                  (static_cast<common_type_t<_Tp, _Up>>(::cuda::std::declval<_Up>()))));

template <class _Tp, class _Up>
_CCCL_CONCEPT _Common_type_constructible = _CCCL_FRAGMENT(__common_type_constructible_, _Tp, _Up);

template <class _Tp, class _Up>
_CCCL_CONCEPT_FRAGMENT(
  __common_with_,
  requires()(
    requires(_Common_type_constructible<_Tp, _Up>),
    requires(same_as<common_type_t<_Tp, _Up>, common_type_t<_Up, _Tp>>),
    requires(common_reference_with<add_lvalue_reference_t<const _Tp>, add_lvalue_reference_t<const _Up>>),
    requires(
      common_reference_with<add_lvalue_reference_t<common_type_t<_Tp, _Up>>,
                            common_reference_t<add_lvalue_reference_t<const _Tp>, add_lvalue_reference_t<const _Up>>>)));

template <class _Tp, class _Up>
_CCCL_CONCEPT common_with = _CCCL_FRAGMENT(__common_with_, _Tp, _Up);

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CONCEPTS_COMMON_WITH_H
