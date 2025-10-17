/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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

// template<class OtherElementType, size_t OtherExtent>
//    constexpr span(const span<OtherElementType, OtherExtent>& s) noexcept;
//
//  Remarks: This constructor shall not participate in overload resolution unless:
//      Extent == dynamic_extent || Extent == OtherExtent is true, and
//      OtherElementType(*)[] is convertible to ElementType(*)[].

#include <uscl/std/cassert>
#include <uscl/std/span>

#include "test_macros.h"

__host__ __device__ void checkCV()
{
  cuda::std::span<int> sp;
  //  cuda::std::span<const          int>  csp;
  cuda::std::span<volatile int> vsp;
  //  cuda::std::span<const volatile int> cvsp;

  cuda::std::span<int, 0> sp0;
  //  cuda::std::span<const          int, 0>  csp0;
  cuda::std::span<volatile int, 0> vsp0;
  //  cuda::std::span<const volatile int, 0> cvsp0;

  //  dynamic -> dynamic
  {
    cuda::std::span<const int> s1{sp}; // a cuda::std::span<const          int> pointing at int.
    cuda::std::span<volatile int> s2{sp}; // a cuda::std::span<      volatile int> pointing at int.
    cuda::std::span<const volatile int> s3{sp}; // a cuda::std::span<const volatile int> pointing at int.
    cuda::std::span<const volatile int> s4{vsp}; // a cuda::std::span<const volatile int> pointing at volatile int.
    assert(s1.size() + s2.size() + s3.size() + s4.size() == 0);
  }

  //  static -> static
  {
    cuda::std::span<const int, 0> s1{sp0}; // a cuda::std::span<const          int> pointing at int.
    cuda::std::span<volatile int, 0> s2{sp0}; // a cuda::std::span<      volatile int> pointing at int.
    cuda::std::span<const volatile int, 0> s3{sp0}; // a cuda::std::span<const volatile int> pointing at int.
    cuda::std::span<const volatile int, 0> s4{vsp0}; // a cuda::std::span<const volatile int> pointing at volatile int.
    assert(s1.size() + s2.size() + s3.size() + s4.size() == 0);
  }

  //  static -> dynamic
  {
    cuda::std::span<const int> s1{sp0}; // a cuda::std::span<const          int> pointing at int.
    cuda::std::span<volatile int> s2{sp0}; // a cuda::std::span<      volatile int> pointing at int.
    cuda::std::span<const volatile int> s3{sp0}; // a cuda::std::span<const volatile int> pointing at int.
    cuda::std::span<const volatile int> s4{vsp0}; // a cuda::std::span<const volatile int> pointing at volatile int.
    assert(s1.size() + s2.size() + s3.size() + s4.size() == 0);
  }

  //  dynamic -> static (not allowed)
}

template <typename T>
__host__ __device__ constexpr bool testConstexprSpan()
{
  cuda::std::span<T> s0{};
  cuda::std::span<T, 0> s1{};
  cuda::std::span<T> s2(s1); // static -> dynamic
  static_assert(noexcept(cuda::std::span<T>{s0}));
  static_assert(noexcept(cuda::std::span<T, 0>{s1}));
  static_assert(noexcept(cuda::std::span<T>{s1}));

  return s0.data() == nullptr && s0.size() == 0 && s1.data() == nullptr && s1.size() == 0 && s2.data() == nullptr
      && s2.size() == 0;
}

template <typename T>
__host__ __device__ void testRuntimeSpan()
{
  cuda::std::span<T> s0{};
  cuda::std::span<T, 0> s1{};
  cuda::std::span<T> s2(s1); // static -> dynamic
  static_assert(noexcept(cuda::std::span<T>{s0}));
  static_assert(noexcept(cuda::std::span<T, 0>{s1}));
  static_assert(noexcept(cuda::std::span<T>{s1}));

  assert(s0.data() == nullptr && s0.size() == 0);
  assert(s1.data() == nullptr && s1.size() == 0);
  assert(s2.data() == nullptr && s2.size() == 0);
}

struct A
{};

int main(int, char**)
{
  static_assert(testConstexprSpan<int>());
  static_assert(testConstexprSpan<long>());
  static_assert(testConstexprSpan<double>());
  static_assert(testConstexprSpan<A>());

  testRuntimeSpan<int>();
  testRuntimeSpan<long>();
  testRuntimeSpan<double>();
  testRuntimeSpan<A>();

  checkCV();

  return 0;
}
