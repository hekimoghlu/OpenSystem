/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 9, 2025.
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
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16

// constexpr subrange() requires default_initializable<I>;

#include <uscl/std/cassert>
#include <uscl/std/cstddef>
#include <uscl/std/ranges>

#include "test_iterators.h"

// An input_or_output_iterator that is not default constructible so we can test
// the `requires` on subrange's default constructor.
struct NoDefaultIterator
{
  using difference_type = cuda::std::ptrdiff_t;
  NoDefaultIterator()   = delete;
  __host__ __device__ NoDefaultIterator& operator++();
  __host__ __device__ void operator++(int);
  __host__ __device__ int& operator*() const;
  __host__ __device__ friend bool operator==(NoDefaultIterator const&, NoDefaultIterator const&);
#if TEST_STD_VER < 2020
  __host__ __device__ friend bool operator!=(NoDefaultIterator const&, NoDefaultIterator const&);
#endif
};
static_assert(cuda::std::input_or_output_iterator<NoDefaultIterator>);

// A sentinel type for the above iterator
struct Sentinel
{
  __host__ __device__ friend bool operator==(NoDefaultIterator const&, Sentinel const&);
  __host__ __device__ friend bool operator==(Sentinel const&, NoDefaultIterator const&);
  __host__ __device__ friend bool operator!=(NoDefaultIterator const&, Sentinel const&);
  __host__ __device__ friend bool operator!=(Sentinel const&, NoDefaultIterator const&);
};

__host__ __device__ constexpr bool test()
{
  {
    static_assert(!cuda::std::is_default_constructible_v<
                  cuda::std::ranges::subrange<NoDefaultIterator, Sentinel, cuda::std::ranges::subrange_kind::sized>>);
    static_assert(!cuda::std::is_default_constructible_v<
                  cuda::std::ranges::subrange<NoDefaultIterator, Sentinel, cuda::std::ranges::subrange_kind::unsized>>);
  }

  {
    using Iter = forward_iterator<int*>;
    cuda::std::ranges::subrange<Iter, Iter, cuda::std::ranges::subrange_kind::sized> subrange;
    assert(subrange.begin() == Iter());
    assert(subrange.end() == Iter());
  }
  {
    using Iter = forward_iterator<int*>;
    cuda::std::ranges::subrange<Iter, Iter, cuda::std::ranges::subrange_kind::unsized> subrange;
    assert(subrange.begin() == Iter());
    assert(subrange.end() == Iter());
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
