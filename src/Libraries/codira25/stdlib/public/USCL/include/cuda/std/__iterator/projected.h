/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 2, 2022.
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
#ifndef _CUDA_STD___ITERATOR_PROJECTED_H
#define _CUDA_STD___ITERATOR_PROJECTED_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__iterator/concepts.h>
#include <uscl/std/__iterator/incrementable_traits.h>
#include <uscl/std/__type_traits/enable_if.h>
#include <uscl/std/__type_traits/remove_cvref.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _It, class _Proj, class = void>
struct __projected_impl
{
  struct __type
  {
    using value_type = remove_cvref_t<indirect_result_t<_Proj, _It>>;
    _CCCL_API inline indirect_result_t<_Proj, _It> operator*() const; // not defined
  };
};

template <class _It, class _Proj>
struct __projected_impl<_It, _Proj, enable_if_t<weakly_incrementable<_It>>>
{
  struct __type
  {
    using value_type      = remove_cvref_t<indirect_result_t<_Proj, _It>>;
    using difference_type = iter_difference_t<_It>;
    _CCCL_API inline indirect_result_t<_Proj, _It> operator*() const; // not defined
  };
};

_CCCL_TEMPLATE(class _It, class _Proj)
_CCCL_REQUIRES(indirectly_readable<_It> _CCCL_AND indirectly_regular_unary_invocable<_Proj, _It>)
using projected = typename __projected_impl<_It, _Proj>::__type;

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ITERATOR_PROJECTED_H
