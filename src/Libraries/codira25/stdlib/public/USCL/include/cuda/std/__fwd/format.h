/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FWD_FORMAT_H
#define _CUDA_STD___FWD_FORMAT_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__fwd/iterator.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _CharT>
class _CCCL_TYPE_VISIBILITY_DEFAULT basic_format_parse_context;

using format_parse_context = basic_format_parse_context<char>;
#if _CCCL_HAS_WCHAR_T()
using wformat_parse_context = basic_format_parse_context<wchar_t>;
#endif // _CCCL_HAS_WCHAR_T()

template <class _CharT>
class _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_PREFERRED_NAME(format_parse_context)
#if _CCCL_HAS_WCHAR_T()
  _CCCL_PREFERRED_NAME(wformat_parse_context)
#endif // _CCCL_HAS_WCHAR_T()
    basic_format_parse_context;

template <class _CharT>
class __fmt_output_buffer;

template <class _Context>
class _CCCL_TYPE_VISIBILITY_DEFAULT basic_format_arg;

template <class _OutIt, class _CharT>
class _CCCL_TYPE_VISIBILITY_DEFAULT basic_format_context;

using format_context = basic_format_context<__back_insert_iterator<__fmt_output_buffer<char>>, char>;
#if _CCCL_HAS_WCHAR_T()
using wformat_context = basic_format_context<__back_insert_iterator<__fmt_output_buffer<wchar_t>>, wchar_t>;
#endif // _CCCL_HAS_WCHAR_T()

template <class _OutIt, class _CharT>
class _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_PREFERRED_NAME(format_context)
#if _CCCL_HAS_WCHAR_T()
  _CCCL_PREFERRED_NAME(wformat_context)
#endif // _CCCL_HAS_WCHAR_T()
    basic_format_context;

template <class _Context>
class _CCCL_TYPE_VISIBILITY_DEFAULT basic_format_args;

using format_args = basic_format_args<format_context>;
#if _CCCL_HAS_WCHAR_T()
using wformat_args = basic_format_args<wformat_context>;
#endif // _CCCL_HAS_WCHAR_T()

template <class _Context>
class _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_PREFERRED_NAME(format_args)
#if _CCCL_HAS_WCHAR_T()
  _CCCL_PREFERRED_NAME(wformat_args)
#endif // _CCCL_HAS_WCHAR_T()
    basic_format_args;

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FWD_FORMAT_H
