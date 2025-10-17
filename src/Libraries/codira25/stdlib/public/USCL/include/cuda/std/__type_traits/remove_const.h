/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 30, 2023.
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

#ifndef _CUDA_STD___TYPE_TRAITS_REMOVE_CONST_H
#define _CUDA_STD___TYPE_TRAITS_REMOVE_CONST_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_REMOVE_CONST) && !defined(_LIBCUDACXX_USE_REMOVE_CONST_FALLBACK)
template <class _Tp>
struct remove_const
{
  using type _CCCL_NODEBUG_ALIAS = _CCCL_BUILTIN_REMOVE_CONST(_Tp);
};

template <class _Tp>
using remove_const_t _CCCL_NODEBUG_ALIAS = _CCCL_BUILTIN_REMOVE_CONST(_Tp);

#else

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_const
{
  using type = _Tp;
};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_const<const _Tp>
{
  using type = _Tp;
};

template <class _Tp>
using remove_const_t _CCCL_NODEBUG_ALIAS = typename remove_const<_Tp>::type;

#endif // defined(_CCCL_BUILTIN_REMOVE_CONST) && !defined(_LIBCUDACXX_USE_REMOVE_CONST_FALLBACK)

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_REMOVE_CONST_H
