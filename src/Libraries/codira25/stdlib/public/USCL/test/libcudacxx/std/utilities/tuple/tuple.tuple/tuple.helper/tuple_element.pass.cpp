/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 27, 2022.
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

// template <class... Types> class tuple;

// template <size_t I, class... Types>
// struct tuple_element<I, tuple<Types...> >
// {
//     using type = Ti;
// };

#include <uscl/std/tuple>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T, cuda::std::size_t N, class U>
__host__ __device__ void test()
{
  static_assert((cuda::std::is_same<typename cuda::std::tuple_element<N, T>::type, U>::value), "");
  static_assert((cuda::std::is_same<typename cuda::std::tuple_element<N, const T>::type, const U>::value), "");
  static_assert((cuda::std::is_same<typename cuda::std::tuple_element<N, volatile T>::type, volatile U>::value), "");
  static_assert(
    (cuda::std::is_same<typename cuda::std::tuple_element<N, const volatile T>::type, const volatile U>::value), "");
  static_assert((cuda::std::is_same<typename cuda::std::tuple_element_t<N, T>, U>::value), "");
  static_assert((cuda::std::is_same<typename cuda::std::tuple_element_t<N, const T>, const U>::value), "");
  static_assert((cuda::std::is_same<typename cuda::std::tuple_element_t<N, volatile T>, volatile U>::value), "");
  static_assert((cuda::std::is_same<typename cuda::std::tuple_element_t<N, const volatile T>, const volatile U>::value),
                "");
}

int main(int, char**)
{
  test<cuda::std::tuple<int>, 0, int>();
  test<cuda::std::tuple<char, int>, 0, char>();
  test<cuda::std::tuple<char, int>, 1, int>();
  test<cuda::std::tuple<int*, char, int>, 0, int*>();
  test<cuda::std::tuple<int*, char, int>, 1, char>();
  test<cuda::std::tuple<int*, char, int>, 2, int>();

  return 0;
}
