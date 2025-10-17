/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 1, 2023.
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

// <cuda/std/type_traits>

// template <class T> struct is_aggregate;
// template <class T> constexpr bool is_aggregate_v = is_aggregate<T>::value;

#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_true()
{
#if defined(_CCCL_BUILTIN_IS_AGGREGATE)
  static_assert(cuda::std::is_aggregate<T>::value, "");
  static_assert(cuda::std::is_aggregate<const T>::value, "");
  static_assert(cuda::std::is_aggregate<volatile T>::value, "");
  static_assert(cuda::std::is_aggregate<const volatile T>::value, "");
  static_assert(cuda::std::is_aggregate_v<T>, "");
  static_assert(cuda::std::is_aggregate_v<const T>, "");
  static_assert(cuda::std::is_aggregate_v<volatile T>, "");
  static_assert(cuda::std::is_aggregate_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__ void test_false()
{
#if defined(_CCCL_BUILTIN_IS_AGGREGATE)
  static_assert(!cuda::std::is_aggregate<T>::value, "");
  static_assert(!cuda::std::is_aggregate<const T>::value, "");
  static_assert(!cuda::std::is_aggregate<volatile T>::value, "");
  static_assert(!cuda::std::is_aggregate<const volatile T>::value, "");
  static_assert(!cuda::std::is_aggregate_v<T>, "");
  static_assert(!cuda::std::is_aggregate_v<const T>, "");
  static_assert(!cuda::std::is_aggregate_v<volatile T>, "");
  static_assert(!cuda::std::is_aggregate_v<const volatile T>, "");
#endif
}

struct Aggregate
{};
struct HasCons
{
  __host__ __device__ HasCons(int);
};
struct HasPriv
{
  __host__ __device__ void PreventUnusedPrivateMemberWarning();

private:
  int x;
};
struct Union
{
  int x;
  void* y;
};

int main(int, char**)
{
  {
    test_false<void>();
    test_false<int>();
    test_false<void*>();
    test_false<void()>();
    test_false<void() const>();
    test_false<void (Aggregate::*)(int) const>();
    test_false<Aggregate&>();
    test_false<HasCons>();
    test_false<HasPriv>();
  }
  {
    test_true<Aggregate>();
    test_true<Aggregate[]>();
    test_true<Aggregate[42][101]>();
    test_true<Union>();
  }

  return 0;
}
