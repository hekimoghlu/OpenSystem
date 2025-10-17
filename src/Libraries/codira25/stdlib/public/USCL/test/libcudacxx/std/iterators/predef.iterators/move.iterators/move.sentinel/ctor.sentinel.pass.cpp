/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// move_sentinel

// constexpr explicit move_sentinel(S s);

#include <uscl/std/cassert>
#include <uscl/std/iterator>

__host__ __device__ constexpr bool test()
{
  // The underlying sentinel is an integer.
  {
    static_assert(!cuda::std::is_convertible_v<int, cuda::std::move_sentinel<int>>);
    cuda::std::move_sentinel<int> m(42);
    assert(m.base() == 42);
  }

  // The underlying sentinel is a pointer.
  {
    static_assert(!cuda::std::is_convertible_v<int*, cuda::std::move_sentinel<int*>>);
    int i = 42;
    cuda::std::move_sentinel<int*> m(&i);
    assert(m.base() == &i);
  }

  // The underlying sentinel is a user-defined type with an explicit default constructor.
  {
    struct S
    {
      explicit S() = default;
      __host__ __device__ constexpr explicit S(int j)
          : i(j)
      {}
      int i = 3;
    };
    static_assert(!cuda::std::is_convertible_v<S, cuda::std::move_sentinel<S>>);
    cuda::std::move_sentinel<S> m(S(42));
    assert(m.base().i == 42);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
