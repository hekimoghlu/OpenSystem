/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 8, 2024.
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

// ~variant();

#include <uscl/std/cassert>
#include <uscl/std/type_traits>
#include <uscl/std/variant>

#include "test_macros.h"

struct NonTDtor
{
  STATIC_MEMBER_VAR(count, int)
  NonTDtor() = default;
  __host__ __device__ ~NonTDtor()
  {
    ++count();
  }
};
static_assert(!cuda::std::is_trivially_destructible<NonTDtor>::value, "");

struct NonTDtor1
{
  STATIC_MEMBER_VAR(count, int)
  NonTDtor1() = default;
  __host__ __device__ ~NonTDtor1()
  {
    ++count();
  }
};
static_assert(!cuda::std::is_trivially_destructible<NonTDtor1>::value, "");

struct TDtor
{
  __host__ __device__ TDtor(const TDtor&) {} // non-trivial copy
  ~TDtor() = default;
};
static_assert(!cuda::std::is_trivially_copy_constructible<TDtor>::value, "");
static_assert(cuda::std::is_trivially_destructible<TDtor>::value, "");

int main(int, char**)
{
  {
    using V = cuda::std::variant<int, long, TDtor>;
    static_assert(cuda::std::is_trivially_destructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<NonTDtor, int, NonTDtor1>;
    static_assert(!cuda::std::is_trivially_destructible<V>::value, "");
    {
      V v(cuda::std::in_place_index<0>);
      assert(NonTDtor::count() == 0);
      assert(NonTDtor1::count() == 0);
    }
    assert(NonTDtor::count() == 1);
    assert(NonTDtor1::count() == 0);
    NonTDtor::count() = 0;
    {
      V v(cuda::std::in_place_index<1>);
    }
    assert(NonTDtor::count() == 0);
    assert(NonTDtor1::count() == 0);
    {
      V v(cuda::std::in_place_index<2>);
      assert(NonTDtor::count() == 0);
      assert(NonTDtor1::count() == 0);
    }
    assert(NonTDtor::count() == 0);
    assert(NonTDtor1::count() == 1);
  }

  return 0;
}
