/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 17, 2025.
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
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// template <class T, size_t N>
// struct array

// Test the size and alignment matches that of an array of a given type.

#include <uscl/std/array>
#include <uscl/std/cstddef>
#include <uscl/std/iterator>
#include <uscl/std/type_traits>

#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4324) // structure was padded due to alignment specifier

template <class T, cuda::std::size_t Size>
struct MyArray
{
  T elems[Size];
};

template <class T, cuda::std::size_t Size>
__host__ __device__ void test()
{
  typedef T CArrayT[Size == 0 ? 1 : Size];
  typedef cuda::std::array<T, Size> ArrayT;
  typedef MyArray<T, Size == 0 ? 1 : Size> MyArrayT;
  static_assert(sizeof(ArrayT) == sizeof(CArrayT), "");
  static_assert(sizeof(ArrayT) == sizeof(MyArrayT), "");
  static_assert(alignof(ArrayT) == alignof(MyArrayT), "");
}

template <class T>
__host__ __device__ void test_type()
{
  test<T, 1>();
  test<T, 42>();
  test<T, 0>();
}

struct alignas(alignof(cuda::std::max_align_t) * 2) TestType1
{};

struct alignas(alignof(cuda::std::max_align_t) * 2) TestType2
{
  char data[1000];
};

struct alignas(alignof(cuda::std::max_align_t)) TestType3
{
  char data[1000];
};

int main(int, char**)
{
  test_type<char>();
  test_type<int>();
  test_type<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_type<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()

  test_type<cuda::std::max_align_t>();
  test_type<TestType1>();
  test_type<TestType2>();
  test_type<TestType3>();

  return 0;
}
