/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 25, 2025.
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

// template <class... Types>
//   class tuple_size<tuple<Types...>>
//     : public integral_constant<size_t, sizeof...(Types)> { };

// XFAIL: gcc-4.8, gcc-4.9

#include <uscl/std/tuple>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T, class = decltype(cuda::std::tuple_size<T>::value)>
__host__ __device__ constexpr bool has_value(int)
{
  return true;
}
template <class>
__host__ __device__ constexpr bool has_value(long)
{
  return false;
}
template <class T>
__host__ __device__ constexpr bool has_value()
{
  return has_value<T>(0);
}

struct Dummy
{};

int main(int, char**)
{
  // Test that the ::value member does not exist
  static_assert(has_value<cuda::std::tuple<int> const>(), "");
  static_assert(has_value<cuda::std::pair<int, long> volatile>(), "");
  static_assert(!has_value<int>(), "");
  static_assert(!has_value<const int>(), "");
  static_assert(!has_value<volatile void>(), "");
  static_assert(!has_value<const volatile cuda::std::tuple<int>&>(), "");

  return 0;
}
