/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>

// template <class T> struct variant_size; // undefined
// template <class T> struct variant_size<const T>;
// template <class T> struct variant_size<volatile T>;
// template <class T> struct variant_size<const volatile T>;
// template <class T> constexpr size_t variant_size_v
//     = variant_size<T>::value;

// #include <uscl/std/memory>
#include <uscl/std/type_traits>
#include <uscl/std/variant>

#include "test_macros.h"

template <class V, size_t E>
__host__ __device__ void test()
{
  static_assert(cuda::std::variant_size<V>::value == E, "");
  static_assert(cuda::std::variant_size<const V>::value == E, "");
  static_assert(cuda::std::variant_size<volatile V>::value == E, "");
  static_assert(cuda::std::variant_size<const volatile V>::value == E, "");
  static_assert(cuda::std::variant_size_v<V> == E, "");
  static_assert(cuda::std::variant_size_v<const V> == E, "");
  static_assert(cuda::std::variant_size_v<volatile V> == E, "");
  static_assert(cuda::std::variant_size_v<const volatile V> == E, "");
  static_assert(
    cuda::std::is_base_of<cuda::std::integral_constant<cuda::std::size_t, E>, cuda::std::variant_size<V>>::value, "");
};

int main(int, char**)
{
  test<cuda::std::variant<>, 0>();
  test<cuda::std::variant<void*>, 1>();
  test<cuda::std::variant<long, long, void*, double>, 4>();

  return 0;
}
