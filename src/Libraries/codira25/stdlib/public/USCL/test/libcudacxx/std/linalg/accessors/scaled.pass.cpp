/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <uscl/std/cassert>
#include <uscl/std/linalg>
#include <uscl/std/type_traits>

__host__ __device__ void constexpr_test()
{
  using T = int;
  using E = cuda::std::extents<size_t, 2>;
  cuda::std::array<T, 2> d{42, 43};
  cuda::std::mdspan<T, E> md(d.data(), E{});
  // operator() type
  {
    [[maybe_unused]] auto scaled_md = cuda::std::linalg::scaled(2.0f, md);

    static_assert(cuda::std::is_same<decltype(scaled_md(0)), float>::value, "wrong type");
  }
  // nested_accessor()
  {
    [[maybe_unused]] auto scaled_md = cuda::std::linalg::scaled(2, md);

    static_assert(
      cuda::std::is_same<decltype(scaled_md.accessor().nested_accessor()), cuda::std::default_accessor<T>>::value,
      "wrong type");
  }
}

__host__ __device__ void runtime_test()
{
  using T = const int;
  using E = cuda::std::extents<size_t, 2>;
  cuda::std::array<T, 2> d{42, 43};
  cuda::std::mdspan<T, E> md(d.data(), 2);
  // operator() value
  {
    auto scaled_md = cuda::std::linalg::scaled(2, md);
    assert(scaled_md(0) == 42 * 2);
    assert(scaled_md(1) == 43 * 2);
  }
  // access()
  {
    auto scaled_md = cuda::std::linalg::scaled(2, md);
    assert(scaled_md.accessor().access(d.data(), 1) == 43 * 2);
  }
  // offset()
  {
    auto scaled_md = cuda::std::linalg::scaled(2, md);
    assert(scaled_md.accessor().offset(d.data(), 1) == d.data() + 1);
  }
  // scaling_factor()
  {
    auto scaled_md = cuda::std::linalg::scaled(2, md);
    assert(scaled_md.accessor().scaling_factor() == 2);
  }
  // composition
  {
    auto scaled_md1 = cuda::std::linalg::scaled(2, md);
    auto scaled_md2 = cuda::std::linalg::scaled(3, scaled_md1);
    assert(scaled_md2(0) == 42 * 2 * 3);
    assert(scaled_md2(1) == 43 * 2 * 3);
  }
  // copy constructor
  {
    auto scaled_md1 = cuda::std::linalg::scaled(2, md);
    auto scaled_md2 = scaled_md1;
    assert(scaled_md2(0) == 42 * 2);
    assert(scaled_md2(1) == 43 * 2);
  }
}

int main(int, char**)
{
  constexpr_test();
  runtime_test();
  return 0;
}
