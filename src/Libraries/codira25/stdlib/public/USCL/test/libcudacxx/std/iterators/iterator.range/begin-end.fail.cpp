/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 2, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "test_macros.h"

// <cuda/std/iterator>
// template <class C> auto begin(C& c) -> decltype(c.begin());
// template <class C> auto begin(const C& c) -> decltype(c.begin());
// template <class C> auto end(C& c) -> decltype(c.end());
// template <class C> auto end(const C& c) -> decltype(c.end());
// template <class E> reverse_iterator<const E*> rbegin(initializer_list<E> il);
// template <class E> reverse_iterator<const E*> rend(initializer_list<E> il);

#include <uscl/std/cassert>
#include <uscl/std/iterator>

namespace Foo
{
struct FakeContainer
{};
typedef int FakeIter;

__host__ __device__ FakeIter begin(const FakeContainer&)
{
  return 1;
}
__host__ __device__ FakeIter end(const FakeContainer&)
{
  return 2;
}
__host__ __device__ FakeIter rbegin(const FakeContainer&)
{
  return 3;
}
__host__ __device__ FakeIter rend(const FakeContainer&)
{
  return 4;
}

__host__ __device__ FakeIter cbegin(const FakeContainer&)
{
  return 11;
}
__host__ __device__ FakeIter cend(const FakeContainer&)
{
  return 12;
}
__host__ __device__ FakeIter crbegin(const FakeContainer&)
{
  return 13;
}
__host__ __device__ FakeIter crend(const FakeContainer&)
{
  return 14;
}
} // namespace Foo

int main(int, char**)
{
  // Bug #28927 - shouldn't find these via ADL
  TEST_IGNORE_NODISCARD cuda::std::cbegin(Foo::FakeContainer());
  TEST_IGNORE_NODISCARD cuda::std::cend(Foo::FakeContainer());
  TEST_IGNORE_NODISCARD cuda::std::crbegin(Foo::FakeContainer());
  TEST_IGNORE_NODISCARD cuda::std::crend(Foo::FakeContainer());

  return 0;
}
#endif
