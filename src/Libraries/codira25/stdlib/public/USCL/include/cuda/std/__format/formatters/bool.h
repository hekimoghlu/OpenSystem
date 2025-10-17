/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 29, 2025.
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

#ifndef _CUDA_STD__FORMAT_FORMATERS_BOOL_H
#define _CUDA_STD__FORMAT_FORMATERS_BOOL_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__format/format_integral.h>
#include <uscl/std/__format/format_spec_parser.h>
#include <uscl/std/__format/formatter.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

//!
//! @brief Formatter for boolean values.
//!
//! @tparam _CharT The character type used for formatting.
//!
template <class _CharT>
struct __fmt_formatter_bool
{
  //!
  //! @brief Parses the formatting specifications for boolean values.
  //!
  //! @param __ctx The parsing context containing the format specification.
  //! @return An iterator pointing to the end of the parsed format specification.
  //!
  template <class _ParseCtx>
  _CCCL_API constexpr typename _ParseCtx::iterator parse(_ParseCtx& __ctx)
  {
    typename _ParseCtx::iterator __result = __parser_.__parse(__ctx, ::cuda::std::__fmt_spec_fields_int());
    ::cuda::std::__fmt_process_parsed_bool(__parser_);
    return __result;
  }

  //!
  //! @brief Formats a boolean value according to the parsed specifications.
  //!
  //! @param __value The boolean value to format.
  //! @param __ctx The formatting context where the formatted output will be stored.
  //! @return An iterator pointing to the end of the formatted output.
  //!
  template <class _FmtCtx>
  _CCCL_API typename _FmtCtx::iterator format(bool __value, _FmtCtx& __ctx) const
  {
    switch (__parser_.__type_)
    {
      case __fmt_spec_type::__default:
      case __fmt_spec_type::__string:
        return ::cuda::std::__fmt_format_bool(__value, __ctx, __parser_.__get_parsed_std_spec(__ctx));
      case __fmt_spec_type::__binary_lower_case:
      case __fmt_spec_type::__binary_upper_case:
      case __fmt_spec_type::__octal:
      case __fmt_spec_type::__decimal:
      case __fmt_spec_type::__hexadecimal_lower_case:
      case __fmt_spec_type::__hexadecimal_upper_case:
        // Promotes bool to an integral type. This reduces the number of
        // instantiations of __format_integer reducing code size.
        return ::cuda::std::__fmt_format_int(
          static_cast<unsigned>(__value), __ctx, __parser_.__get_parsed_std_spec(__ctx));
      default:
        _CCCL_UNREACHABLE();
    }
  }

private:
  __fmt_spec_parser<_CharT> __parser_; //!< The parser for format specifications.
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<bool, char> : __fmt_formatter_bool<char>
{};

#if _CCCL_HAS_WCHAR_T()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter<bool, wchar_t> : __fmt_formatter_bool<wchar_t>
{};
#endif // _CCCL_HAS_WCHAR_T()

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD__FORMAT_FORMATERS_BOOL_H
