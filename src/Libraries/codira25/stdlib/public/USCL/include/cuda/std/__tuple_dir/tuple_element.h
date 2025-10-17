/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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

#ifndef _CUDA_STD___TUPLE_TUPLE_ELEMENT_H
#define _CUDA_STD___TUPLE_TUPLE_ELEMENT_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__tuple_dir/tuple_indices.h>
#include <uscl/std/__tuple_dir/tuple_types.h>
#include <uscl/std/__type_traits/add_const.h>
#include <uscl/std/__type_traits/add_cv.h>
#include <uscl/std/__type_traits/add_volatile.h>
#include <uscl/std/__type_traits/type_list.h>
#include <uscl/std/cstddef>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element;

template <size_t _Ip, class _Tp>
using tuple_element_t _CCCL_NODEBUG_ALIAS = typename tuple_element<_Ip, _Tp>::type;

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, const _Tp>
{
  using type _CCCL_NODEBUG_ALIAS = const tuple_element_t<_Ip, _Tp>;
};

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, volatile _Tp>
{
  using type _CCCL_NODEBUG_ALIAS = volatile tuple_element_t<_Ip, _Tp>;
};

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, const volatile _Tp>
{
  using type _CCCL_NODEBUG_ALIAS = const volatile tuple_element_t<_Ip, _Tp>;
};

template <size_t _Ip, class... _Types>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, __tuple_types<_Types...>>
{
  static_assert(_Ip < sizeof...(_Types), "tuple_element index out of range");
  using type _CCCL_NODEBUG_ALIAS = __type_index_c<_Ip, _Types...>;
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TUPLE_ELEMENT_H
