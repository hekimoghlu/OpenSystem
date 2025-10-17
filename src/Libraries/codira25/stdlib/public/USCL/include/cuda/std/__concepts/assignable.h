/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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

#ifndef _CUDA_STD___CONCEPTS_ASSIGNABLE_H
#define _CUDA_STD___CONCEPTS_ASSIGNABLE_H

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
#include <uscl/std/__type_traits/is_reference.h>
#include <uscl/std/__type_traits/make_const_lvalue_ref.h>
#include <uscl/std/__utility/forward.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_HAS_CONCEPTS()

// [concept.assignable]

template <class _Lhs, class _Rhs>
concept assignable_from =
  is_lvalue_reference_v<_Lhs> && common_reference_with<__make_const_lvalue_ref<_Lhs>, __make_const_lvalue_ref<_Rhs>>
  && requires(_Lhs __lhs, _Rhs&& __rhs) {
       { __lhs = ::cuda::std::forward<_Rhs>(__rhs) } -> same_as<_Lhs>;
     };

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Lhs, class _Rhs>
_CCCL_CONCEPT_FRAGMENT(
  __assignable_from_,
  requires(_Lhs __lhs,
           _Rhs&& __rhs)(requires(is_lvalue_reference_v<_Lhs>),
                         requires(common_reference_with<__make_const_lvalue_ref<_Lhs>, __make_const_lvalue_ref<_Rhs>>),
                         requires(same_as<_Lhs, decltype(__lhs = ::cuda::std::forward<_Rhs>(__rhs))>)));

template <class _Lhs, class _Rhs>
_CCCL_CONCEPT assignable_from = _CCCL_FRAGMENT(__assignable_from_, _Lhs, _Rhs);

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CONCEPTS_ASSIGNABLE_H
