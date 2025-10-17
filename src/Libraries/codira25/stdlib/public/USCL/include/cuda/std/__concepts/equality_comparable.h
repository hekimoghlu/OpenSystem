/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 7, 2025.
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

#ifndef _CUDA_STD___CONCEPTS_EQUALITY_COMPARABLE_H
#define _CUDA_STD___CONCEPTS_EQUALITY_COMPARABLE_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__concepts/boolean_testable.h>
#include <uscl/std/__concepts/common_reference_with.h>
#include <uscl/std/__concepts/concept_macros.h>
#include <uscl/std/__type_traits/common_reference.h>
#include <uscl/std/__type_traits/make_const_lvalue_ref.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_HAS_CONCEPTS()

// [concept.equalitycomparable]

template <class _Tp, class _Up>
concept __weakly_equality_comparable_with =
  requires(__make_const_lvalue_ref<_Tp> __t, __make_const_lvalue_ref<_Up> __u) {
    { __t == __u } -> __boolean_testable;
    { __t != __u } -> __boolean_testable;
    { __u == __t } -> __boolean_testable;
    { __u != __t } -> __boolean_testable;
  };

template <class _Tp>
concept equality_comparable = __weakly_equality_comparable_with<_Tp, _Tp>;

template <class _Tp, class _Up>
concept equality_comparable_with =
  equality_comparable<_Tp> && equality_comparable<_Up>
  && common_reference_with<__make_const_lvalue_ref<_Tp>, __make_const_lvalue_ref<_Up>>
  && equality_comparable<common_reference_t<__make_const_lvalue_ref<_Tp>, __make_const_lvalue_ref<_Up>>>
  && __weakly_equality_comparable_with<_Tp, _Up>;

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__with_lvalue_reference_, requires()(typename(__make_const_lvalue_ref<_Tp>)));

template <class _Tp>
_CCCL_CONCEPT _With_lvalue_reference = _CCCL_FRAGMENT(__with_lvalue_reference_, _Tp);

template <class _Tp, class _Up>
_CCCL_CONCEPT_FRAGMENT(
  __weakly_equality_comparable_with_,
  requires(__make_const_lvalue_ref<_Tp> __t, __make_const_lvalue_ref<_Up> __u)(
    requires(_With_lvalue_reference<_Tp>),
    requires(_With_lvalue_reference<_Up>),
    requires(__boolean_testable<decltype(__t == __u)>),
    requires(__boolean_testable<decltype(__t != __u)>),
    requires(__boolean_testable<decltype(__u == __t)>),
    requires(__boolean_testable<decltype(__u != __t)>)));

template <class _Tp, class _Up>
_CCCL_CONCEPT __weakly_equality_comparable_with = _CCCL_FRAGMENT(__weakly_equality_comparable_with_, _Tp, _Up);

template <class _Tp>
_CCCL_CONCEPT equality_comparable = __weakly_equality_comparable_with<_Tp, _Tp>;

template <class _Tp, class _Up>
_CCCL_CONCEPT_FRAGMENT(
  __equality_comparable_with_,
  requires()(
    requires(equality_comparable<_Tp>),
    requires(equality_comparable<_Up>),
    requires(common_reference_with<__make_const_lvalue_ref<_Tp>, __make_const_lvalue_ref<_Up>>),
    requires(equality_comparable<common_reference_t<__make_const_lvalue_ref<_Tp>, __make_const_lvalue_ref<_Up>>>),
    requires(__weakly_equality_comparable_with<_Tp, _Up>)));

template <class _Tp, class _Up>
_CCCL_CONCEPT equality_comparable_with = _CCCL_FRAGMENT(__equality_comparable_with_, _Tp, _Up);

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CONCEPTS_EQUALITY_COMPARABLE_H
