/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 20, 2022.
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

#include <uscl/std/cassert>
#include <uscl/std/cstddef>
#include <uscl/std/type_traits>

#include "test_macros.h"

// typedef decltype(nullptr) nullptr_t;

struct A
{
  __host__ __device__ A(cuda::std::nullptr_t) {}
};

template <class T>
__host__ __device__ void test_conversions()
{
  {
    // GCC spuriously claims that p is unused when T is nullptr_t, probably due to optimizations?
    [[maybe_unused]] T p = 0;
    assert(p == nullptr);
  }
  {
    // GCC spuriously claims that p is unused when T is nullptr_t, probably due to optimizations?
    [[maybe_unused]] T p = nullptr;
    assert(p == nullptr);
    assert(nullptr == p);
    assert(!(p != nullptr));
    assert(!(nullptr != p));
  }
}

template <class T>
struct Voider
{
  typedef void type;
};
template <class T, class = void>
struct has_less : cuda::std::false_type
{};

template <class T>
struct has_less<T, typename Voider<decltype(cuda::std::declval<T>() < nullptr)>::type> : cuda::std::true_type
{};

template <class T>
__host__ __device__ void test_comparisons()
{
  // GCC spuriously claims that p is unused, probably due to optimizations?
  [[maybe_unused]] T p = nullptr;
  assert(p == nullptr);
  assert(!(p != nullptr));
  assert(nullptr == p);
  assert(!(nullptr != p));
}

TEST_DIAG_SUPPRESS_CLANG("-Wnull-conversion")
__host__ __device__ void test_nullptr_conversions()
{
// GCC does not accept this due to CWG Defect #1423
// http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1423
#if TEST_COMPILER(CLANG) && !TEST_CUDA_COMPILER(NVCC) && !TEST_CUDA_COMPILER(CLANG)
  {
    bool b = nullptr;
    assert(!b);
  }
#endif // TEST_COMPILER(CLANG) && !TEST_CUDA_COMPILER(NVCC) && !TEST_CUDA_COMPILER(CLANG)
  {
    bool b(nullptr);
    assert(!b);
  }
}

int main(int, char**)
{
  static_assert(sizeof(cuda::std::nullptr_t) == sizeof(void*), "sizeof(cuda::std::nullptr_t) == sizeof(void*)");

  {
    test_conversions<cuda::std::nullptr_t>();
    test_conversions<void*>();
    test_conversions<A*>();
    test_conversions<void (*)()>();
    test_conversions<void (A::*)()>();
    test_conversions<int A::*>();
  }
  {
    // TODO Enable this assertion when all compilers implement core DR 583.
    // static_assert(!has_less<cuda::std::nullptr_t>::value, "");
    test_comparisons<cuda::std::nullptr_t>();
    test_comparisons<void*>();
    test_comparisons<A*>();
    test_comparisons<void (*)()>();
  }
  test_nullptr_conversions();

  return 0;
}
