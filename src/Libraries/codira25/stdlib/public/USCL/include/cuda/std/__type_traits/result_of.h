/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 22, 2023.
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

#ifndef _CUDA_STD___TYPE_TRAITS_RESULT_OF_H
#define _CUDA_STD___TYPE_TRAITS_RESULT_OF_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__functional/invoke.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// result_of

#if _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_TYPE_TRAITS)
template <class _Callable>
class result_of;

template <class _Fp, class... _Args>
class _LIBCUDACXX_DEPRECATED _CCCL_TYPE_VISIBILITY_DEFAULT result_of<_Fp(_Args...)> : public __invoke_of<_Fp, _Args...>
{};

template <class _Tp>
using result_of_t _LIBCUDACXX_DEPRECATED = typename result_of<_Tp>::type;
#endif // _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_TYPE_TRAITS)

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_RESULT_OF_H
