/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 26, 2025.
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
//
//  LWG #2212 says that tuple_size and tuple_element must be
//     available after including <utility>

#include <uscl/std/tuple>
// cuda::std::array not supported
// #include <uscl/std/array>
#include <uscl/std/type_traits>

#include "test_macros.h"

// cuda::std::array not supported
/*
template <class T, cuda::std::size_t N, class U, size_t idx>
__host__ __device__ void test()
{
    static_assert((cuda::std::is_base_of<cuda::std::integral_constant<cuda::std::size_t, N>,
                                   cuda::std::tuple_size<T> >::value), "");
    static_assert((cuda::std::is_base_of<cuda::std::integral_constant<cuda::std::size_t, N>,
                                   cuda::std::tuple_size<const T> >::value), "");
    static_assert((cuda::std::is_base_of<cuda::std::integral_constant<cuda::std::size_t, N>,
                                   cuda::std::tuple_size<volatile T> >::value), "");
    static_assert((cuda::std::is_base_of<cuda::std::integral_constant<cuda::std::size_t, N>,
                                   cuda::std::tuple_size<const volatile T> >::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<idx, T>::type, U>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<idx, const T>::type, const U>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<idx, volatile T>::type, volatile U>::value),
""); static_assert((cuda::std::is_same<typename cuda::std::tuple_element<idx, const volatile T>::type, const volatile
U>::value), "");
}
*/

int main(int, char**)
{
  // cuda::std::array not supported
  /*
  test<cuda::std::array<int, 5>, 5, int, 0>();
  test<cuda::std::array<int, 5>, 5, int, 1>();
  test<cuda::std::array<const char *, 4>, 4, const char *, 3>();
  test<cuda::std::array<volatile int, 4>, 4, volatile int, 3>();
  test<cuda::std::array<char *, 3>, 3, char *, 1>();
  test<cuda::std::array<char *, 3>, 3, char *, 2>();
  */

  return 0;
}
