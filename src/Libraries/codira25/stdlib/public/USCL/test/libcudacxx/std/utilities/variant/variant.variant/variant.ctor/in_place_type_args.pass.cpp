/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 24, 2023.
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

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>

// template <class ...Types> class variant;

// template <class Tp, class ...Args>
// constexpr explicit variant(in_place_type_t<Tp>, Args&&...);

#include <uscl/std/cassert>
#include <uscl/std/type_traits>
#include <uscl/std/variant>

#include "test_convertible.h"
#include "test_macros.h"

__host__ __device__ void test_ctor_sfinae()
{
  {
    using V = cuda::std::variant<int>;
    static_assert(cuda::std::is_constructible<V, cuda::std::in_place_type_t<int>, int>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_type_t<int>, int>(), "");
  }
  {
    using V = cuda::std::variant<int, long, long long>;
    static_assert(cuda::std::is_constructible<V, cuda::std::in_place_type_t<long>, int>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_type_t<long>, int>(), "");
  }
  {
    using V = cuda::std::variant<int, long, int*>;
    static_assert(cuda::std::is_constructible<V, cuda::std::in_place_type_t<int*>, int*>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_type_t<int*>, int*>(), "");
  }
  { // duplicate type
    using V = cuda::std::variant<int, long, int>;
    static_assert(!cuda::std::is_constructible<V, cuda::std::in_place_type_t<int>, int>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_type_t<int>, int>(), "");
  }
  { // args not convertible to type
    using V = cuda::std::variant<int, long, int*>;
    static_assert(!cuda::std::is_constructible<V, cuda::std::in_place_type_t<int>, int*>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_type_t<int>, int*>(), "");
  }
  { // type not in variant
    using V = cuda::std::variant<int, long, int*>;
    static_assert(!cuda::std::is_constructible<V, cuda::std::in_place_type_t<long long>, int>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_type_t<long long>, int>(), "");
  }
}

__host__ __device__ void test_ctor_basic()
{
  {
    constexpr cuda::std::variant<int> v(cuda::std::in_place_type<int>, 42);
    static_assert(v.index() == 0, "");
    static_assert(cuda::std::get<0>(v) == 42, "");
  }
  {
    constexpr cuda::std::variant<int, long> v(cuda::std::in_place_type<long>, 42);
    static_assert(v.index() == 1, "");
    static_assert(cuda::std::get<1>(v) == 42, "");
  }
  {
    constexpr cuda::std::variant<int, const int, long> v(cuda::std::in_place_type<const int>, 42);
    static_assert(v.index() == 1, "");
    static_assert(cuda::std::get<1>(v) == 42, "");
  }
  {
    using V = cuda::std::variant<const int, volatile int, int>;
    int x   = 42;
    V v(cuda::std::in_place_type<const int>, x);
    assert(v.index() == 0);
    assert(cuda::std::get<0>(v) == x);
  }
  {
    using V = cuda::std::variant<const int, volatile int, int>;
    int x   = 42;
    V v(cuda::std::in_place_type<volatile int>, x);
    assert(v.index() == 1);
    assert(cuda::std::get<1>(v) == x);
  }
  {
    using V = cuda::std::variant<const int, volatile int, int>;
    int x   = 42;
    V v(cuda::std::in_place_type<int>, x);
    assert(v.index() == 2);
    assert(cuda::std::get<2>(v) == x);
  }
}

int main(int, char**)
{
  test_ctor_basic();
  test_ctor_sfinae();

  return 0;
}
