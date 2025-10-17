/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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

// constexpr bool ends_with(basic_string_view x) const noexcept;

// constexpr bool ends_with(charT x) const noexcept;

// constexpr bool ends_with(const charT* x) const;

#include <uscl/std/cassert>
#include <uscl/std/string_view>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "literal.h"

template <class SV>
__host__ __device__ constexpr void test_ends_with()
{
  using CharT = typename SV::value_type;

  // constexpr bool ends_with(basic_string_view x) const noexcept;

  static_assert(cuda::std::is_same_v<bool, decltype(SV{}.ends_with(SV{}))>);
  static_assert(noexcept(SV{}.ends_with(SV{})));

  {
    const CharT* str = TEST_STRLIT(CharT, "abcde");

    SV a0{};
    SV a1{str + 4, 1};
    SV a2{str + 3, 2};
    SV aNot{TEST_STRLIT(CharT, "def")};

    SV b0{};
    SV b1{str + 4, 1};
    SV b2{str + 3, 2};
    SV b3{str + 2, 3};
    SV b4{str + 1, 4};
    SV b5{str, 5};
    SV bNot{TEST_STRLIT(CharT, "def")};

    assert(a0.ends_with(b0));
    assert(!a0.ends_with(b1));

    assert(a1.ends_with(b0));
    assert(a1.ends_with(b1));
    assert(!a1.ends_with(b2));
    assert(!a1.ends_with(b3));
    assert(!a1.ends_with(b4));
    assert(!a1.ends_with(b5));
    assert(!a1.ends_with(bNot));

    assert(a2.ends_with(b0));
    assert(a2.ends_with(b1));
    assert(a2.ends_with(b2));
    assert(!a2.ends_with(b3));
    assert(!a2.ends_with(b4));
    assert(!a2.ends_with(b5));
    assert(!a2.ends_with(bNot));

    assert(aNot.ends_with(b0));
    assert(!aNot.ends_with(b1));
    assert(!aNot.ends_with(b2));
    assert(!aNot.ends_with(b3));
    assert(!aNot.ends_with(b4));
    assert(!aNot.ends_with(b5));
    assert(aNot.ends_with(bNot));
  }

  // constexpr bool ends_with(charT x) const noexcept;

  static_assert(cuda::std::is_same_v<bool, decltype(SV{}.ends_with(CharT{}))>);
  static_assert(noexcept(SV{}.ends_with(CharT{})));

  {
    SV sv1{};
    SV sv2{TEST_STRLIT(CharT, "abcde")};

    assert(!sv1.ends_with(TEST_CHARLIT(CharT, 'e')));
    assert(!sv1.ends_with(TEST_CHARLIT(CharT, 'x')));
    assert(sv2.ends_with(TEST_CHARLIT(CharT, 'e')));
    assert(!sv2.ends_with(TEST_CHARLIT(CharT, 'x')));
  }

  // constexpr bool ends_with(const charT* x) const;

  static_assert(cuda::std::is_same_v<bool, decltype(SV{}.ends_with(cuda::std::declval<const CharT*>()))>);
  static_assert(noexcept(SV{}.ends_with(cuda::std::declval<const CharT*>())));

  {
    const CharT* str = TEST_STRLIT(CharT, "abcde");

    SV sv0{};
    SV sv1{str + 4, 1};
    SV sv2{str + 3, 2};
    SV svNot{TEST_STRLIT(CharT, "def")};

    assert(sv0.ends_with(TEST_STRLIT(CharT, "")));
    assert(!sv0.ends_with(TEST_STRLIT(CharT, "e")));

    assert(sv1.ends_with(TEST_STRLIT(CharT, "")));
    assert(sv1.ends_with(TEST_STRLIT(CharT, "e")));
    assert(!sv1.ends_with(TEST_STRLIT(CharT, "de")));
    assert(!sv1.ends_with(TEST_STRLIT(CharT, "cde")));
    assert(!sv1.ends_with(TEST_STRLIT(CharT, "bcde")));
    assert(!sv1.ends_with(TEST_STRLIT(CharT, "abcde")));
    assert(!sv1.ends_with(TEST_STRLIT(CharT, "def")));

    assert(sv2.ends_with(TEST_STRLIT(CharT, "")));
    assert(sv2.ends_with(TEST_STRLIT(CharT, "e")));
    assert(sv2.ends_with(TEST_STRLIT(CharT, "de")));
    assert(!sv2.ends_with(TEST_STRLIT(CharT, "cde")));
    assert(!sv2.ends_with(TEST_STRLIT(CharT, "bcde")));
    assert(!sv2.ends_with(TEST_STRLIT(CharT, "abcde")));
    assert(!sv2.ends_with(TEST_STRLIT(CharT, "def")));

    assert(svNot.ends_with(TEST_STRLIT(CharT, "")));
    assert(!svNot.ends_with(TEST_STRLIT(CharT, "e")));
    assert(!svNot.ends_with(TEST_STRLIT(CharT, "de")));
    assert(!svNot.ends_with(TEST_STRLIT(CharT, "cde")));
    assert(!svNot.ends_with(TEST_STRLIT(CharT, "bcde")));
    assert(!svNot.ends_with(TEST_STRLIT(CharT, "abcde")));
    assert(svNot.ends_with(TEST_STRLIT(CharT, "def")));
  }
}

__host__ __device__ constexpr bool test()
{
  test_ends_with<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_ends_with<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_ends_with<cuda::std::u16string_view>();
  test_ends_with<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_ends_with<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
