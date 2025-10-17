/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 25, 2022.
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

// UNSUPPORTED: msvc

// <utility>

// template <class T1, class T2> struct pair

// template<class U, class V> pair& operator=(pair<U, V>&& p);

#include <uscl/std/utility>
// cuda/std/memory not supported
// #include <uscl/std/memory>
#include <uscl/std/cassert>

#include "archetypes.h"
#include "test_macros.h"

struct Base
{
  __host__ __device__ virtual ~Base() {}
};

struct Derived : public Base
{};

int main(int, char**)
{
  // cuda/std/memory not supported
  /*
  {
      typedef cuda::std::pair<cuda::std::unique_ptr<Derived>, short> P1;
      typedef cuda::std::pair<cuda::std::unique_ptr<Base>, long> P2;
      P1 p1(cuda::std::unique_ptr<Derived>(), static_cast<short>(4));
      P2 p2;
      p2 = cuda::std::move(p1);
      assert(p2.first == nullptr);
      assert(p2.second == 4);
  }
  */
  {
    using C = TestTypes::TestType;
    using P = cuda::std::pair<int, C>;
    using T = cuda::std::pair<long, C>;
    T t(42, -42);
    P p(101, 101);
    C::reset_constructors();
    p = cuda::std::move(t);
    assert(C::constructed() == 0);
    assert(C::assigned() == 1);
    assert(C::copy_assigned() == 0);
    assert(C::move_assigned() == 1);
    assert(p.first == 42);
    assert(p.second.value == -42);
  }

  return 0;
}
