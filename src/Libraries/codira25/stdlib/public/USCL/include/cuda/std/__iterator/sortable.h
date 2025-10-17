/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 19, 2024.
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

#ifndef _CUDA_STD___ITERATOR_SORTABLE_H
#define _CUDA_STD___ITERATOR_SORTABLE_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__functional/identity.h>
#include <uscl/std/__functional/ranges_operations.h>
#include <uscl/std/__iterator/concepts.h>
#include <uscl/std/__iterator/permutable.h>
#include <uscl/std/__iterator/projected.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_HAS_CONCEPTS()

template <class _Iter, class _Comp = ::cuda::std::ranges::less, class _Proj = identity>
concept sortable = permutable<_Iter> && indirect_strict_weak_order<_Comp, projected<_Iter, _Proj>>;

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Iter, class _Comp, class _Proj>
_CCCL_CONCEPT_FRAGMENT(
  __sortable_,
  requires()(requires(permutable<_Iter>), requires(indirect_strict_weak_order<_Comp, projected<_Iter, _Proj>>)));

template <class _Iter, class _Comp = ::cuda::std::ranges::less, class _Proj = identity>
_CCCL_CONCEPT sortable = _CCCL_FRAGMENT(__sortable_, _Iter, _Comp, _Proj);

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ITERATOR_SORTABLE_H
