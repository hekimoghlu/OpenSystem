/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 25, 2022.
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

// <numeric>

// Became constexpr in C++20
// template <InputIterator Iter, MoveConstructible T,
//           Callable<auto, const T&, Iter::reference> BinaryOperation>
//   requires HasAssign<T, BinaryOperation::result_type>
//         && CopyConstructible<BinaryOperation>
//   T
//   accumulate(Iter first, Iter last, T init, BinaryOperation binary_op);

#include <uscl/std/cassert>
#include <uscl/std/functional>
#include <uscl/std/numeric>
#ifdef _LIBCUDACXX_HAS_STRING
#  include <cuda/std/string>
#endif // _LIBCUDACXX_HAS_STRING

#include "test_iterators.h"
#include "test_macros.h"

struct rvalue_addable
{
  bool correctOperatorUsed = false;

  // make sure the predicate is passed an rvalue and an lvalue (so check that the first argument was moved)
  __host__ __device__ constexpr rvalue_addable operator()(rvalue_addable&& r, rvalue_addable const&)
  {
    r.correctOperatorUsed = true;
    return cuda::std::move(r);
  }
};

__host__ __device__ constexpr rvalue_addable operator+(rvalue_addable& lhs, rvalue_addable const&)
{
  lhs.correctOperatorUsed = false;
  return lhs;
}

__host__ __device__ constexpr rvalue_addable operator+(rvalue_addable&& lhs, rvalue_addable const&)
{
  lhs.correctOperatorUsed = true;
  return cuda::std::move(lhs);
}

__host__ __device__ constexpr void test_use_move()
{
  rvalue_addable arr[100];
  auto res1 = cuda::std::accumulate(arr, arr + 100, rvalue_addable());
  auto res2 = cuda::std::accumulate(arr, arr + 100, rvalue_addable(), /*predicate=*/rvalue_addable());
  assert(res1.correctOperatorUsed);
  assert(res2.correctOperatorUsed);
}

#ifdef _LIBCUDACXX_HAS_STRING
__host__ __device__ constexpr void test_string()
{
  cuda::std::string sa[] = {"a", "b", "c"};
  assert(cuda::std::accumulate(sa, sa + 3, cuda::std::string()) == "abc");
  assert(cuda::std::accumulate(sa, sa + 3, cuda::std::string(), cuda::std::plus<cuda::std::string>()) == "abc");
}
#endif // _LIBCUDACXX_HAS_STRING

template <class Iter, class T>
__host__ __device__ constexpr void test(Iter first, Iter last, T init, T x)
{
  assert(cuda::std::accumulate(first, last, init, cuda::std::multiplies<T>()) == x);
}

template <class Iter>
__host__ __device__ constexpr void test()
{
  int ia[]    = {1, 2, 3, 4, 5, 6};
  unsigned sa = sizeof(ia) / sizeof(ia[0]);
  test(Iter(ia), Iter(ia), 1, 1);
  test(Iter(ia), Iter(ia), 10, 10);
  test(Iter(ia), Iter(ia + 1), 1, 1);
  test(Iter(ia), Iter(ia + 1), 10, 10);
  test(Iter(ia), Iter(ia + 2), 1, 2);
  test(Iter(ia), Iter(ia + 2), 10, 20);
  test(Iter(ia), Iter(ia + sa), 1, 720);
  test(Iter(ia), Iter(ia + sa), 10, 7200);
}

__host__ __device__ constexpr bool test()
{
  test<cpp17_input_iterator<const int*>>();
  test<forward_iterator<const int*>>();
  test<bidirectional_iterator<const int*>>();
  test<random_access_iterator<const int*>>();
  test<const int*>();

  test_use_move();

#ifdef _LIBCUDACXX_HAS_STRING
  test_string();
#endif // _LIBCUDACXX_HAS_STRING

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
