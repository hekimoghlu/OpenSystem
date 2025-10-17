/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 11, 2022.
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

// <cuda/std/iterator>

// move_iterator

// explicit move_iterator(Iter i);
//
//  constexpr in C++17

#include <uscl/std/cassert>
#include <uscl/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class It>
__host__ __device__ constexpr bool test()
{
  static_assert(cuda::std::is_constructible<cuda::std::move_iterator<It>, const It&>::value, "");
  static_assert(cuda::std::is_constructible<cuda::std::move_iterator<It>, It&&>::value, "");
  static_assert(!cuda::std::is_convertible<const It&, cuda::std::move_iterator<It>>::value, "");
  static_assert(!cuda::std::is_convertible<It&&, cuda::std::move_iterator<It>>::value, "");

  char s[] = "123";
  {
    It it = It(s);
    cuda::std::move_iterator<It> r(it);
    assert(base(r.base()) == s);
  }
  {
    It it = It(s);
    cuda::std::move_iterator<It> r(cuda::std::move(it));
    assert(base(r.base()) == s);
  }
  return true;
}

template <class It>
__host__ __device__ constexpr bool test_moveonly()
{
  static_assert(!cuda::std::is_constructible<cuda::std::move_iterator<It>, const It&>::value, "");
  static_assert(cuda::std::is_constructible<cuda::std::move_iterator<It>, It&&>::value, "");
  static_assert(!cuda::std::is_convertible<const It&, cuda::std::move_iterator<It>>::value, "");
  static_assert(!cuda::std::is_convertible<It&&, cuda::std::move_iterator<It>>::value, "");

  char s[] = "123";
  {
    It it = It(s);
    cuda::std::move_iterator<It> r(cuda::std::move(it));
    assert(base(r.base()) == s);
  }
  return true;
}

int main(int, char**)
{
  test<cpp17_input_iterator<char*>>();
  test<forward_iterator<char*>>();
  test<bidirectional_iterator<char*>>();
  test<random_access_iterator<char*>>();
  test<char*>();
  test<const char*>();

  static_assert(test<cpp17_input_iterator<char*>>(), "");
  static_assert(test<forward_iterator<char*>>(), "");
  static_assert(test<bidirectional_iterator<char*>>(), "");
  static_assert(test<random_access_iterator<char*>>(), "");
  static_assert(test<char*>(), "");
  static_assert(test<const char*>(), "");

  test<contiguous_iterator<char*>>();
  test_moveonly<cpp20_input_iterator<char*>>();
  static_assert(test<contiguous_iterator<char*>>(), "");
  static_assert(test_moveonly<cpp20_input_iterator<char*>>(), "");

  return 0;
}
