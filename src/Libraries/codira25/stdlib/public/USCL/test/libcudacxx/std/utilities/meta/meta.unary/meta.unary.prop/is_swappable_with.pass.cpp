/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 18, 2023.
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

// is_swappable_with

#include <uscl/std/type_traits>
// NOTE: This header is not currently supported by libcu++.
// #include <uscl/std/vector>
#include "test_macros.h"

namespace MyNS
{

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

struct C
{};
struct D
{};

__host__ __device__ void swap(A&, A&) {}

__host__ __device__ void swap(A&, B&) {}
__host__ __device__ void swap(B&, A&) {}

__host__ __device__ void swap(A&, C&) {} // missing swap(C, A)
__host__ __device__ void swap(D&, C&) {}

struct M
{};

__host__ __device__ void swap(M&&, M&&) {}

} // namespace MyNS

int main(int, char**)
{
  using namespace MyNS;
  {
    // Test that is_swappable_with doesn't apply an lvalue reference
    // to the type. Instead it is up to the user.
    static_assert(!cuda::std::is_swappable_with<int, int>::value, "");
    static_assert(cuda::std::is_swappable_with<int&, int&>::value, "");
    static_assert(cuda::std::is_swappable_with<M, M>::value, "");
    static_assert(cuda::std::is_swappable_with<A&, A&>::value, "");
  }
  {
    // test that heterogeneous swap is allowed only if both 'swap(A, B)' and
    // 'swap(B, A)' are valid.
    static_assert(cuda::std::is_swappable_with<A&, B&>::value, "");
    static_assert(!cuda::std::is_swappable_with<A&, C&>::value, "");
    static_assert(!cuda::std::is_swappable_with<D&, C&>::value, "");
  }
  {
    // test that cv void is guarded against as required.
    static_assert(!cuda::std::is_swappable_with_v<void, int>, "");
    static_assert(!cuda::std::is_swappable_with_v<int, void>, "");
    static_assert(!cuda::std::is_swappable_with_v<const void, const volatile void>, "");
  }
  {
    // test for presence of is_swappable_with_v
    static_assert(cuda::std::is_swappable_with_v<int&, int&>, "");
    static_assert(!cuda::std::is_swappable_with_v<D&, C&>, "");
  }

  return 0;
}
