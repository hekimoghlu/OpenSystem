/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 3, 2024.
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
//
//===----------------------------------------------------------------------===//

// XFAIL: gcc-4

// <utility>

// template <class T1, class T2> struct pair

// template<class U, class V> pair(U&& x, V&& y);

#include <uscl/std/utility>
// cuda/std/memory not supported
// #include <uscl/std/memory>
#include <uscl/std/cassert>

#include "archetypes.h"
#include "test_convertible.h"
#include "test_macros.h"
using namespace ImplicitTypes; // Get implicitly archetypes

template <class T1, class T1Arg, bool CanCopy = true, bool CanConvert = CanCopy>
__host__ __device__ void test_sfinae()
{
  using P1 = cuda::std::pair<T1, int>;
  using P2 = cuda::std::pair<int, T1>;
  using T2 = int const&;
  static_assert(cuda::std::is_constructible<P1, T1Arg, T2>::value == CanCopy, "");
  static_assert(test_convertible<P1, T1Arg, T2>() == CanConvert, "");
  static_assert(cuda::std::is_constructible<P2, T2, T1Arg>::value == CanCopy, "");
  static_assert(test_convertible<P2, T2, T1Arg>() == CanConvert, "");
}

struct ExplicitT
{
  __host__ __device__ constexpr explicit ExplicitT(int x)
      : value(x)
  {}
  int value;
};

struct ImplicitT
{
  __host__ __device__ constexpr ImplicitT(int x)
      : value(x)
  {}
  int value;
};

int main(int, char**)
{
  // cuda/std/memory not supported
  /*
  {
      typedef cuda::std::pair<cuda::std::unique_ptr<int>, short*> P;
      P p(cuda::std::unique_ptr<int>(new int(3)), nullptr);
      assert(*p.first == 3);
      assert(p.second == nullptr);
  }
  */
  {
    // Test non-const lvalue and rvalue types
    test_sfinae<AllCtors, AllCtors&>();
    test_sfinae<AllCtors, AllCtors&&>();
    test_sfinae<ExplicitTypes::AllCtors, ExplicitTypes::AllCtors&, true, false>();
    test_sfinae<ExplicitTypes::AllCtors, ExplicitTypes::AllCtors&&, true, false>();
    test_sfinae<CopyOnly, CopyOnly&>();
    test_sfinae<CopyOnly, CopyOnly&&>();
    test_sfinae<ExplicitTypes::CopyOnly, ExplicitTypes::CopyOnly&, true, false>();
    test_sfinae<ExplicitTypes::CopyOnly, ExplicitTypes::CopyOnly&&, true, false>();
    test_sfinae<MoveOnly, MoveOnly&, false>();
    test_sfinae<MoveOnly, MoveOnly&&>();
    test_sfinae<ExplicitTypes::MoveOnly, ExplicitTypes::MoveOnly&, false>();
    test_sfinae<ExplicitTypes::MoveOnly, ExplicitTypes::MoveOnly&&, true, false>();
    test_sfinae<NonCopyable, NonCopyable&, false>();
    test_sfinae<NonCopyable, NonCopyable&&, false>();
    test_sfinae<ExplicitTypes::NonCopyable, ExplicitTypes::NonCopyable&, false>();
    test_sfinae<ExplicitTypes::NonCopyable, ExplicitTypes::NonCopyable&&, false>();
  }
  {
    // Test converting types
    test_sfinae<ConvertingType, int&>();
    test_sfinae<ConvertingType, const int&>();
    test_sfinae<ConvertingType, int&&>();
    test_sfinae<ConvertingType, const int&&>();
    test_sfinae<ExplicitTypes::ConvertingType, int&, true, false>();
    test_sfinae<ExplicitTypes::ConvertingType, const int&, true, false>();
    test_sfinae<ExplicitTypes::ConvertingType, int&&, true, false>();
    test_sfinae<ExplicitTypes::ConvertingType, const int&&, true, false>();
  }
  { // explicit constexpr test
    constexpr cuda::std::pair<ExplicitT, ExplicitT> p(42, 43);
    static_assert(p.first.value == 42, "");
    static_assert(p.second.value == 43, "");
  }
  { // implicit constexpr test
    constexpr cuda::std::pair<ImplicitT, ImplicitT> p = {42, 43};
    static_assert(p.first.value == 42, "");
    static_assert(p.second.value == 43, "");
  }

  return 0;
}
