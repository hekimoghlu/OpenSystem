/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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

// constexpr explicit
// basic_format_parse_context(basic_string_view<charT> fmt,
//                            size_t num_args = 0) noexcept

#include <uscl/std/__format_>
#include <uscl/std/cassert>
#include <uscl/std/string_view>
#include <uscl/std/type_traits>

#include "literal.h"

template <class CharT>
__host__ __device__ constexpr void test()
{
  // Validate the constructor is explicit.
  static_assert(
    !cuda::std::is_convertible_v<cuda::std::basic_string_view<CharT>, cuda::std::basic_format_parse_context<CharT>>);
  static_assert(!cuda::std::is_copy_constructible_v<cuda::std::basic_format_parse_context<CharT>>);
  static_assert(!cuda::std::is_copy_assignable_v<cuda::std::basic_format_parse_context<CharT>>);
  // The move operations are implicitly deleted due to the
  // deleted copy operations.
  static_assert(!cuda::std::is_move_constructible_v<cuda::std::basic_format_parse_context<CharT>>);
  static_assert(!cuda::std::is_move_assignable_v<cuda::std::basic_format_parse_context<CharT>>);

  static_assert(noexcept(cuda::std::basic_format_parse_context{cuda::std::basic_string_view<CharT>{}}));
  static_assert(noexcept(cuda::std::basic_format_parse_context{cuda::std::basic_string_view<CharT>{}, 42}));

  constexpr const CharT* fmt = TEST_STRLIT(CharT, "abc");

  {
    cuda::std::basic_format_parse_context<CharT> context(fmt);
    assert(cuda::std::to_address(context.begin()) == &fmt[0]);
    assert(cuda::std::to_address(context.end()) == &fmt[3]);
  }
  {
    cuda::std::basic_string_view<CharT> view{fmt};
    cuda::std::basic_format_parse_context<CharT> context(view);
    assert(context.begin() == view.begin());
    assert(context.end() == view.end());
  }
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

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
