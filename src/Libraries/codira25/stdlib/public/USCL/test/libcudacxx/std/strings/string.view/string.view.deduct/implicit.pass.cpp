/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 28, 2025.
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

// <cuda/std/string_view>

// basic_string_view

// Make sure that the implicitly-generated CTAD works.

#include <uscl/std/string_view>
#include <uscl/std/type_traits>

#include "literal.h"

template <class CharT>
__host__ __device__ constexpr void test_implicit_ctad()
{
  const CharT* str = TEST_STRLIT(CharT, "Hello world!");
  cuda::std::basic_string_view sv{str};
  static_assert(cuda::std::is_same_v<decltype(sv), cuda::std::basic_string_view<CharT>>);
}

__host__ __device__ constexpr bool test()
{
  test_implicit_ctad<char>();
#if _CCCL_HAS_CHAR8_T()
  test_implicit_ctad<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test_implicit_ctad<char16_t>();
  test_implicit_ctad<char32_t>();
#if _CCCL_HAS_WCHAR_T()
  test_implicit_ctad<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
