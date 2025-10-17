/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 28, 2023.
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

// template<semiregular S>
//   class move_sentinel;
#include <uscl/std/concepts>
#include <uscl/std/iterator>

#include "test_iterators.h"

__host__ __device__ void test()
{
  // Pointer.
  {
    using It = int*;
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sized_sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      !cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
  }

  // `Cpp17InputIterator`.
  {
    using It = cpp17_input_iterator<int*>;
    static_assert(
      cuda::std::sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      !cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
  }

  // `cuda::std::input_iterator`.
  {
    using It = cpp20_input_iterator<int*>;
    static_assert(
      cuda::std::sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      !cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
  }

  // `cuda::std::forward_iterator`.
  {
    using It = forward_iterator<int*>;
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(!cuda::std::sized_sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      !cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
  }

  // `cuda::std::bidirectional_iterator`.
  {
    using It = bidirectional_iterator<int*>;
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(!cuda::std::sized_sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      !cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
  }

  // `cuda::std::random_access_iterator`.
  {
    using It = random_access_iterator<int*>;
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sized_sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      !cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
  }

  // `cuda::std::contiguous_iterator`.
  {
    using It = contiguous_iterator<int*>;
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sized_sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      !cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
  }

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  // `cuda::std::contiguous_iterator` with the spaceship operator.
  {
    using It = three_way_contiguous_iterator<int*>;
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sized_sentinel_for<cuda::std::move_sentinel<It>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      !cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sentinel_wrapper<It>>, cuda::std::move_iterator<It>>);
    static_assert(cuda::std::sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
    static_assert(
      cuda::std::sized_sentinel_for<cuda::std::move_sentinel<sized_sentinel<It>>, cuda::std::move_iterator<It>>);
  }
#endif
}

int main(int, char**)
{
  return 0;
}
