/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 12, 2025.
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
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FUNCTIONAL_BINARY_FUNCTION_H
#define _CUDA_STD___FUNCTIONAL_BINARY_FUNCTION_H

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

#if defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)

template <class _Arg1, class _Arg2, class _Result>
struct _CCCL_TYPE_VISIBILITY_DEFAULT _LIBCUDACXX_DEPRECATED binary_function
{
  using first_argument_type  = _Arg1;
  using second_argument_type = _Arg2;
  using result_type          = _Result;
};

#endif // defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)

template <class _Arg1, class _Arg2, class _Result>
struct __binary_function_keep_layout_base
{
#if _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
  using first_argument_type _LIBCUDACXX_DEPRECATED  = _Arg1;
  using second_argument_type _LIBCUDACXX_DEPRECATED = _Arg2;
  using result_type _LIBCUDACXX_DEPRECATED          = _Result;
#endif // _LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS
};

#if defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
_CCCL_SUPPRESS_DEPRECATED_PUSH
template <class _Arg1, class _Arg2, class _Result>
using __binary_function = binary_function<_Arg1, _Arg2, _Result>;
_CCCL_SUPPRESS_DEPRECATED_POP
#else
template <class _Arg1, class _Arg2, class _Result>
using __binary_function = __binary_function_keep_layout_base<_Arg1, _Arg2, _Result>;
#endif // !_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FUNCTIONAL_BINARY_FUNCTION_H
