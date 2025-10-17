/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_IN_FUN_RESULT_H
#define _CUDA_STD___ALGORITHM_IN_FUN_RESULT_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__concepts/convertible_to.h>
#include <uscl/std/__utility/move.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_RANGES

template <class _InIter1, class _Func1>
struct in_fun_result
{
  _CCCL_NO_UNIQUE_ADDRESS _InIter1 in;
  _CCCL_NO_UNIQUE_ADDRESS _Func1 fun;

  _CCCL_TEMPLATE(class _InIter2, class _Func2)
  _CCCL_REQUIRES(convertible_to<const _InIter1&, _InIter2> _CCCL_AND convertible_to<const _Func1&, _Func2>)
  _CCCL_API constexpr operator in_fun_result<_InIter2, _Func2>() const&
  {
    return {in, fun};
  }

  _CCCL_TEMPLATE(class _InIter2, class _Func2)
  _CCCL_REQUIRES(convertible_to<_InIter1, _InIter2> _CCCL_AND convertible_to<_Func1, _Func2>)
  _CCCL_API constexpr operator in_fun_result<_InIter2, _Func2>() &&
  {
    return {::cuda::std::move(in), ::cuda::std::move(fun)};
  }
};

_CCCL_END_NAMESPACE_RANGES

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_IN_FUN_RESULT_H
