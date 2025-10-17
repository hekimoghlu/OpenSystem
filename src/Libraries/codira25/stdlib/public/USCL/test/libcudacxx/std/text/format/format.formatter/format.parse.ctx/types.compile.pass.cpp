/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 28, 2024.
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

// <cuda/std/format>

// Class typedefs:
// template<class charT>
// class basic_format_parse_context {
// public:
//   using char_type = charT;
//   using const_iterator = typename basic_string_view<charT>::const_iterator;
//   using iterator = const_iterator;
// }
//
// Namespace std typedefs:
// using format_parse_context = basic_format_parse_context<char>;
// using wformat_parse_context = basic_format_parse_context<wchar_t>;

#include <uscl/std/__format_>
#include <uscl/std/string_view>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class CharT>
__host__ __device__ constexpr void test()
{
  static_assert(cuda::std::is_same_v<typename cuda::std::basic_format_parse_context<CharT>::char_type, CharT>);
  static_assert(cuda::std::is_same_v<typename cuda::std::basic_format_parse_context<CharT>::const_iterator,
                                     typename cuda::std::basic_string_view<CharT>::const_iterator>);
  static_assert(cuda::std::is_same_v<typename cuda::std::basic_format_parse_context<CharT>::iterator,
                                     typename cuda::std::basic_format_parse_context<CharT>::const_iterator>);
}

__host__ __device__ constexpr bool test()
{
  test<char>();
#if _CCCL_HAS_CHAR8_T()
  test<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test<char16_t>();
  test<char32_t>();
#if _CCCL_HAS_WCHAR_T()
  test<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

static_assert(cuda::std::is_same_v<cuda::std::format_parse_context, cuda::std::basic_format_parse_context<char>>);
#if _CCCL_HAS_WCHAR_T()
static_assert(cuda::std::is_same_v<cuda::std::wformat_parse_context, cuda::std::basic_format_parse_context<wchar_t>>);
#endif // _CCCL_HAS_WCHAR_T()

int main(int, char**)
{
  static_assert(test());
  return 0;
}
