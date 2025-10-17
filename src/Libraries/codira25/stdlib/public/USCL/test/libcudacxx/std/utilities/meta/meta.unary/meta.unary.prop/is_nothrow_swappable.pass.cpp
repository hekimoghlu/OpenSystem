/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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

// type_traits

// is_swappable

#include <uscl/std/type_traits>
// NOTE: This header is not currently supported by libcu++.
// #include <uscl/std/vector>
#include "test_macros.h"

namespace MyNS
{

// Make the test types non-copyable so that generic cuda::std::swap is not valid.
struct A
{
  A(A const&)            = delete;
  A& operator=(A const&) = delete;
};

struct B
{
  B(B const&)            = delete;
  B& operator=(B const&) = delete;
};

__host__ __device__ void swap(A&, A&) noexcept {}
__host__ __device__ void swap(B&, B&) {}

struct M
{
  M(M const&)            = delete;
  M& operator=(M const&) = delete;
};

__host__ __device__ void swap(M&&, M&&) noexcept {}

struct ThrowingMove
{
  __host__ __device__ ThrowingMove(ThrowingMove&&) {}
  __host__ __device__ ThrowingMove& operator=(ThrowingMove&&)
  {
    return *this;
  }
};

} // namespace MyNS

int main(int, char**)
{
  using namespace MyNS;
  {
    // Test that is_swappable applies an lvalue reference to the type.
    static_assert(cuda::std::is_nothrow_swappable<int>::value, "");
    static_assert(cuda::std::is_nothrow_swappable<int&>::value, "");
    static_assert(!cuda::std::is_nothrow_swappable<M>::value, "");
    static_assert(!cuda::std::is_nothrow_swappable<M&&>::value, "");
  }
  {
    // Test that it correctly deduces the noexcept of swap.
    static_assert(cuda::std::is_nothrow_swappable<A>::value, "");
    static_assert(!cuda::std::is_nothrow_swappable<B>::value && cuda::std::is_swappable<B>::value, "");
    static_assert(!cuda::std::is_nothrow_swappable<ThrowingMove>::value && cuda::std::is_swappable<ThrowingMove>::value,
                  "");
  }
  {
    // Test that it doesn't drop the qualifiers
    static_assert(!cuda::std::is_nothrow_swappable<const A>::value, "");
  }
  {
    // test non-referenceable types
    static_assert(!cuda::std::is_nothrow_swappable<void>::value, "");
    static_assert(!cuda::std::is_nothrow_swappable<int() const>::value, "");
    static_assert(!cuda::std::is_nothrow_swappable<int(int, ...) const&>::value, "");
  }
  {
    // test for presence of is_nothrow_swappable_v
    static_assert(cuda::std::is_nothrow_swappable_v<int>, "");
    static_assert(!cuda::std::is_nothrow_swappable_v<void>, "");
  }

  return 0;
}
