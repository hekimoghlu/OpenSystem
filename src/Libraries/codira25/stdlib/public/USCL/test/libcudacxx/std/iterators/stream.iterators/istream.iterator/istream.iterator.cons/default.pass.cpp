/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 7, 2025.
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

// class istream_iterator

// constexpr istream_iterator();
// C++17 says: If is_trivially_default_constructible_v<T> is true, then this
//    constructor is a constexpr constructor.

#include <uscl/std/cassert>
#include <uscl/std/iterator>
#if defined(_LIBCUDACXX_HAS_STRING)
#  include <cuda/std/string>

#  include "test_macros.h"

struct S
{
  S();
}; // not constexpr

template <typename T, bool isTrivial = cuda::std::is_trivially_default_constructible_v<T>>
struct test_trivial
{
  void operator()() const
  {
    [[maybe_unused]] constexpr cuda::std::istream_iterator<T> it;
  }
};

template <typename T>
struct test_trivial<T, false>
{
  void operator()() const {}
};

int main(int, char**)
{
  {
    typedef cuda::std::istream_iterator<int> T;
    T it;
    assert(it == T());
    [[maybe_unused]] constexpr T it2;
  }

  test_trivial<int>()();
  test_trivial<char>()();
  test_trivial<double>()();
  test_trivial<S>()();
  test_trivial<cuda::std::string>()();

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
