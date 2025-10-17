/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 23, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_MAX_H
#define _CUDA_STD___ALGORITHM_MAX_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__algorithm/comp.h>
#include <uscl/std/__algorithm/comp_ref_type.h>
#include <uscl/std/__algorithm/max_element.h>
#include <uscl/std/initializer_list>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Compare>
[[nodiscard]] _CCCL_API constexpr const _Tp& max(const _Tp& __a, const _Tp& __b, _Compare __comp)
{
  return __comp(__a, __b) ? __b : __a;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp>
[[nodiscard]] _CCCL_API constexpr const _Tp& max(const _Tp& __a, const _Tp& __b)
{
  return __a < __b ? __b : __a;
}

template <class _Tp, class _Compare>
[[nodiscard]] _CCCL_API constexpr _Tp max(initializer_list<_Tp> __t, _Compare __comp)
{
  return *::cuda::std::__max_element<__comp_ref_type<_Compare>>(__t.begin(), __t.end(), __comp);
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp max(initializer_list<_Tp> __t)
{
  return *::cuda::std::max_element(__t.begin(), __t.end(), __less{});
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_MAX_H
