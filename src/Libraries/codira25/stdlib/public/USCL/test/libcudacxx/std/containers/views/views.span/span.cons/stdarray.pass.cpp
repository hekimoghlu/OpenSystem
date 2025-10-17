/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 9, 2022.
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

// <span>

// template<size_t N>
//     constexpr span(array<value_type, N>& arr) noexcept;
// template<size_t N>
//     constexpr span(const array<value_type, N>& arr) noexcept;
//
// Remarks: These constructors shall not participate in overload resolution unless:
//   â€” extent == dynamic_extent || N == extent is true, and
//   â€” remove_pointer_t<decltype(data(arr))>(*)[] is convertible to ElementType(*)[].
//

#include <uscl/std/array>
#include <uscl/std/cassert>
#include <uscl/std/span>

#include "test_macros.h"

__host__ __device__ void checkCV()
{
  cuda::std::array<int, 3> arr = {1, 2, 3};
  //  STL says these are not cromulent
  //  std::array<const int,3> carr = {4,5,6};
  //  std::array<volatile int, 3> varr = {7,8,9};
  //  std::array<const volatile int, 3> cvarr = {1,3,5};

  //  Types the same (dynamic sized)
  {
    cuda::std::span<int> s1{arr}; // a cuda::std::span<               int> pointing at int.
  }

  //  Types the same (static sized)
  {
    cuda::std::span<int, 3> s1{arr}; // a cuda::std::span<               int> pointing at int.
  }

  //  types different (dynamic sized)
  {
    cuda::std::span<const int> s1{arr}; // a cuda::std::span<const          int> pointing at int.
    cuda::std::span<volatile int> s2{arr}; // a cuda::std::span<      volatile int> pointing at int.
    cuda::std::span<volatile int> s3{arr}; // a cuda::std::span<      volatile int> pointing at const int.
    cuda::std::span<const volatile int> s4{arr}; // a cuda::std::span<const volatile int> pointing at int.
  }

  //  types different (static sized)
  {
    cuda::std::span<const int, 3> s1{arr}; // a cuda::std::span<const          int> pointing at int.
    cuda::std::span<volatile int, 3> s2{arr}; // a cuda::std::span<      volatile int> pointing at int.
    cuda::std::span<volatile int, 3> s3{arr}; // a cuda::std::span<      volatile int> pointing at const int.
    cuda::std::span<const volatile int, 3> s4{arr}; // a cuda::std::span<const volatile int> pointing at int.
  }
}

template <typename T, typename U>
__host__ __device__ constexpr bool testConstructorArray()
{
  cuda::std::array<U, 2> val = {U(), U()};
  static_assert(noexcept(cuda::std::span<T>{val}));
  static_assert(noexcept(cuda::std::span<T, 2>{val}));
  cuda::std::span<T> s1{val};
  cuda::std::span<T, 2> s2{val};
  return s1.data() == &val[0] && s1.size() == 2 && s2.data() == &val[0] && s2.size() == 2;
}

template <typename T, typename U>
__host__ __device__ constexpr bool testConstructorConstArray()
{
  const cuda::std::array<U, 2> val = {U(), U()};
  static_assert(noexcept(cuda::std::span<const T>{val}));
  static_assert(noexcept(cuda::std::span<const T, 2>{val}));
  cuda::std::span<const T> s1{val};
  cuda::std::span<const T, 2> s2{val};
  return s1.data() == &val[0] && s1.size() == 2 && s2.data() == &val[0] && s2.size() == 2;
}

template <typename T>
__host__ __device__ constexpr bool testConstructors()
{
  static_assert((testConstructorArray<T, T>()));
  static_assert((testConstructorArray<const T, const T>()));
  static_assert((testConstructorArray<const T, T>()));
  static_assert((testConstructorConstArray<T, T>()));
  static_assert((testConstructorConstArray<const T, const T>()));
  static_assert((testConstructorConstArray<const T, T>()));

  return testConstructorArray<T, T>() && testConstructorArray<const T, const T>() && testConstructorArray<const T, T>()
      && testConstructorConstArray<T, T>() && testConstructorConstArray<const T, const T>()
      && testConstructorConstArray<const T, T>();
}

struct A
{};

int main(int, char**)
{
  assert(testConstructors<int>());
  assert(testConstructors<long>());
  assert(testConstructors<double>());
  assert(testConstructors<A>());

  assert(testConstructors<int*>());
  assert(testConstructors<const int*>());

  checkCV();

  return 0;
}
