/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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

// <cuda/std/tuple>

// template<class... Types>
//     tuple<Types&&...> forward_as_tuple(Types&&... t);

#include <uscl/std/cassert>
#include <uscl/std/tuple>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class Tuple>
__host__ __device__ void test0(const Tuple&)
{
  static_assert(cuda::std::tuple_size<Tuple>::value == 0, "");
}

template <class Tuple>
__host__ __device__ void test1a(const Tuple& t)
{
  static_assert(cuda::std::tuple_size<Tuple>::value == 1, "");
  static_assert(cuda::std::is_same<typename cuda::std::tuple_element<0, Tuple>::type, int&&>::value, "");
  assert(cuda::std::get<0>(t) == 1);
}

template <class Tuple>
__host__ __device__ void test1b(const Tuple& t)
{
  static_assert(cuda::std::tuple_size<Tuple>::value == 1, "");
  static_assert(cuda::std::is_same<typename cuda::std::tuple_element<0, Tuple>::type, int&>::value, "");
  assert(cuda::std::get<0>(t) == 2);
}

template <class Tuple>
__host__ __device__ void test2a(const Tuple& t)
{
  static_assert(cuda::std::tuple_size<Tuple>::value == 2, "");
  static_assert(cuda::std::is_same<typename cuda::std::tuple_element<0, Tuple>::type, double&>::value, "");
  static_assert(cuda::std::is_same<typename cuda::std::tuple_element<1, Tuple>::type, char&>::value, "");
  assert(cuda::std::get<0>(t) == 2.5);
  assert(cuda::std::get<1>(t) == 'a');
}

template <class Tuple>
__host__ __device__ constexpr int test3(const Tuple&)
{
  return cuda::std::tuple_size<Tuple>::value;
}

int main(int, char**)
{
  {
    test0(cuda::std::forward_as_tuple());
  }
  {
    test1a(cuda::std::forward_as_tuple(1));
  }
  {
    int i = 2;
    test1b(cuda::std::forward_as_tuple(i));
  }
  {
    double i = 2.5;
    char c   = 'a';
    test2a(cuda::std::forward_as_tuple(i, c));
    static_assert(test3(cuda::std::forward_as_tuple(i, c)) == 2, "");
  }

  return 0;
}
