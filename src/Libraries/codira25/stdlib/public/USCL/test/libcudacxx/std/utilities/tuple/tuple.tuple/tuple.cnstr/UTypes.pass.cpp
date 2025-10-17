/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 19, 2025.
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

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class... UTypes>
//   explicit tuple(UTypes&&... u);

// XFAIL: gcc-4.8, gcc-4.9
// XFAIL: msvc-19.12, msvc-19.13

#include <uscl/std/cassert>
#include <uscl/std/tuple>

#include "MoveOnly.h"
#include "test_convertible.h"
#include "test_macros.h"

struct Empty
{};
struct A
{
  int id_;
  __host__ __device__ explicit constexpr A(int i)
      : id_(i)
  {}
};

struct NoDefault
{
  NoDefault() = delete;
};

// Make sure the _Up... constructor SFINAEs out when the types that
// are not explicitly initialized are not all default constructible.
// Otherwise, cuda::std::is_constructible would return true but instantiating
// the constructor would fail.
__host__ __device__ void test_default_constructible_extension_sfinae()
{
  {
    using Tuple = cuda::std::tuple<MoveOnly, NoDefault>;

    static_assert(!cuda::std::is_constructible<Tuple, MoveOnly>::value, "");

    static_assert(cuda::std::is_constructible<Tuple, MoveOnly, NoDefault>::value, "");
  }
  {
    using Tuple = cuda::std::tuple<MoveOnly, MoveOnly, NoDefault>;

    static_assert(!cuda::std::is_constructible<Tuple, MoveOnly, MoveOnly>::value, "");

    static_assert(cuda::std::is_constructible<Tuple, MoveOnly, MoveOnly, NoDefault>::value, "");
  }
  {
    // Same idea as above but with a nested tuple type.
    using Tuple       = cuda::std::tuple<MoveOnly, NoDefault>;
    using NestedTuple = cuda::std::tuple<MoveOnly, Tuple, MoveOnly, MoveOnly>;

    static_assert(!cuda::std::is_constructible<NestedTuple, MoveOnly, MoveOnly, MoveOnly, MoveOnly>::value, "");

    static_assert(cuda::std::is_constructible<NestedTuple, MoveOnly, Tuple, MoveOnly, MoveOnly>::value, "");
  }
  // testing extensions
#ifdef _LIBCUDACXX_VERSION
  {
    using Tuple       = cuda::std::tuple<MoveOnly, int>;
    using NestedTuple = cuda::std::tuple<MoveOnly, Tuple, MoveOnly, MoveOnly>;

    static_assert(cuda::std::is_constructible<NestedTuple, MoveOnly, MoveOnly, MoveOnly, MoveOnly>::value, "");

    static_assert(cuda::std::is_constructible<NestedTuple, MoveOnly, Tuple, MoveOnly, MoveOnly>::value, "");
  }
#endif
}

int main(int, char**)
{
  {
    cuda::std::tuple<MoveOnly> t(MoveOnly(0));
    assert(cuda::std::get<0>(t) == 0);
  }
  {
    cuda::std::tuple<MoveOnly, MoveOnly> t(MoveOnly(0), MoveOnly(1));
    assert(cuda::std::get<0>(t) == 0);
    assert(cuda::std::get<1>(t) == 1);
  }
  {
    cuda::std::tuple<MoveOnly, MoveOnly, MoveOnly> t(MoveOnly(0), MoveOnly(1), MoveOnly(2));
    assert(cuda::std::get<0>(t) == 0);
    assert(cuda::std::get<1>(t) == 1);
    assert(cuda::std::get<2>(t) == 2);
  }
  // extensions, MSVC issues
#if defined(_LIBCUDACXX_VERSION) && !TEST_COMPILER(MSVC)
  {
    using E   = MoveOnly;
    using Tup = cuda::std::tuple<E, E, E>;
    // Test that the reduced arity initialization extension is only
    // allowed on the explicit constructor.
    static_assert(test_convertible<Tup, E, E, E>(), "");

    Tup t(E(0), E(1));
    static_assert(!test_convertible<Tup, E, E>(), "");
    assert(cuda::std::get<0>(t) == 0);
    assert(cuda::std::get<1>(t) == 1);
    assert(cuda::std::get<2>(t) == MoveOnly());

    Tup t2(E(0));
    static_assert(!test_convertible<Tup, E>(), "");
    assert(cuda::std::get<0>(t2) == 0);
    assert(cuda::std::get<1>(t2) == E());
    assert(cuda::std::get<2>(t2) == E());
  }
#endif
  {
    [[maybe_unused]] constexpr cuda::std::tuple<Empty> t0{Empty()};
  }
  {
    constexpr cuda::std::tuple<A, A> t(3, 2);
    static_assert(cuda::std::get<0>(t).id_ == 3, "");
  }
  // Check that SFINAE is properly applied with the default reduced arity
  // constructor extensions.
  test_default_constructible_extension_sfinae();

  return 0;
}
