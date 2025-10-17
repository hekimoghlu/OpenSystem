/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 6, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ITERATOR_INDIRECTLY_COMPARABLE_H
#define _CUDA_STD___ITERATOR_INDIRECTLY_COMPARABLE_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__functional/identity.h>
#include <uscl/std/__iterator/concepts.h>
#include <uscl/std/__iterator/projected.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_HAS_CONCEPTS()

template <class _Iter1, class _Iter2, class _BinaryPred, class _Proj1 = identity, class _Proj2 = identity>
concept indirectly_comparable =
  indirect_binary_predicate<_BinaryPred, projected<_Iter1, _Proj1>, projected<_Iter2, _Proj2>>;

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Iter1, class _Iter2, class _BinaryPred, class _Proj1, class _Proj2>
_CCCL_CONCEPT_FRAGMENT(
  __indirectly_comparable_,
  requires()(requires(indirect_binary_predicate<_BinaryPred, projected<_Iter1, _Proj1>, projected<_Iter2, _Proj2>>)));

template <class _Iter1, class _Iter2, class _BinaryPred, class _Proj1 = identity, class _Proj2 = identity>
_CCCL_CONCEPT indirectly_comparable =
  _CCCL_FRAGMENT(__indirectly_comparable_, _Iter1, _Iter2, _BinaryPred, _Proj1, _Proj2);

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ITERATOR_INDIRECTLY_COMPARABLE_H
