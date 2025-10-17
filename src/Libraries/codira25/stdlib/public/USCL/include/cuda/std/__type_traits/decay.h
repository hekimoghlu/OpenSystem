/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 11, 2025.
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

#ifndef _CUDA_STD___TYPE_TRAITS_DECAY_H
#define _CUDA_STD___TYPE_TRAITS_DECAY_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__type_traits/add_pointer.h>
#include <uscl/std/__type_traits/conditional.h>
#include <uscl/std/__type_traits/is_array.h>
#include <uscl/std/__type_traits/is_function.h>
#include <uscl/std/__type_traits/is_referenceable.h>
#include <uscl/std/__type_traits/remove_cv.h>
#include <uscl/std/__type_traits/remove_extent.h>
#include <uscl/std/__type_traits/remove_reference.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_DECAY) && !defined(_LIBCUDACXX_USE_DECAY_FALLBACK)
template <class _Tp>
struct decay
{
  using type _CCCL_NODEBUG_ALIAS = _CCCL_BUILTIN_DECAY(_Tp);
};

template <class _Tp>
using decay_t _CCCL_NODEBUG_ALIAS = _CCCL_BUILTIN_DECAY(_Tp);

#else // ^^^ _CCCL_BUILTIN_DECAY ^^^ / vvv !_CCCL_BUILTIN_DECAY vvv

template <class _Up, bool>
struct __decay_impl
{
  using type _CCCL_NODEBUG_ALIAS = remove_cv_t<_Up>;
};

template <class _Up>
struct __decay_impl<_Up, true>
{
public:
  using type _CCCL_NODEBUG_ALIAS =
    conditional_t<is_array<_Up>::value,
                  remove_extent_t<_Up>*,
                  conditional_t<is_function<_Up>::value, add_pointer_t<_Up>, remove_cv_t<_Up>>>;
};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT decay
{
private:
  using _Up _CCCL_NODEBUG_ALIAS = remove_reference_t<_Tp>;

public:
  using type _CCCL_NODEBUG_ALIAS = typename __decay_impl<_Up, __cccl_is_referenceable<_Up>::value>::type;
};

template <class _Tp>
using decay_t _CCCL_NODEBUG_ALIAS = typename decay<_Tp>::type;

#endif // !_CCCL_BUILTIN_DECAY

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_DECAY_H
