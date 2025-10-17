/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 8, 2024.
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

// During the review D130295 it was noticed libc++'s implementation uses
// cuda::std::common_type. When users specialize this template for their own types the
// comparisons would fail. This tests with a specialized cuda::std::common_type.

// <cuda/std/string_view>

#include <uscl/std/cassert>
#include <uscl/std/cstring>
#include <uscl/std/string_view>

#include "test_comparisons.h"

struct char_wrapper
{
  char c;
};

template <>
struct cuda::std::char_traits<char_wrapper>
{
  using char_type = char_wrapper;

  __host__ __device__ static bool eq(char_wrapper lhs, char_wrapper rhs)
  {
    return lhs.c == rhs.c;
  }

  __host__ __device__ static cuda::std::size_t length(const char_wrapper* a)
  {
    static_assert(sizeof(char_wrapper) == 1, "strlen requires this");
    return cuda::std::strlen(reinterpret_cast<const char*>(a));
  }

  __host__ __device__ static int compare(const char_wrapper* lhs, const char_wrapper* rhs, cuda::std::size_t count)
  {
    return cuda::std::char_traits<char>::compare(
      reinterpret_cast<const char*>(lhs), reinterpret_cast<const char*>(rhs), count);
  }
};

using WrappedSV = cuda::std::basic_string_view<char_wrapper, cuda::std::char_traits<char_wrapper>>;

// cuda::std::common_type can be specialized and not have a typedef-name member type.
template <>
struct cuda::std::common_type<WrappedSV, WrappedSV>
{};

struct convertible_to_string_view
{
  WrappedSV sv;
  __host__ __device__ convertible_to_string_view(const char_wrapper* a)
      : sv(a)
  {}
  __host__ __device__ operator WrappedSV() const
  {
    return sv;
  }
};

template <class T, class U>
__host__ __device__ void test()
{
  char_wrapper a[] = {{'a'}, {'b'}, {'c'}, {'\0'}};

  assert((testComparisons(T(a), U(a), true, false)));

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  assert((testOrder(T(a), U(a), cuda::std::weak_ordering::equivalent)));
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
}

int main(int, char**)
{
  test<WrappedSV, convertible_to_string_view>();

  return 0;
}
