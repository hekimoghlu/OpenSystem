/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 13, 2024.
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

// implicitly generated array constructors / assignment operators

#include <uscl/std/array>
#include <uscl/std/cassert>
#include <uscl/std/type_traits>

#include "test_macros.h"

struct NoDefault
{
  __host__ __device__ constexpr NoDefault(int) {}
};

struct NonTrivialCopy
{
  __host__ __device__ constexpr NonTrivialCopy() {}
  __host__ __device__ constexpr NonTrivialCopy(NonTrivialCopy const&) {}
  __host__ __device__ constexpr NonTrivialCopy& operator=(NonTrivialCopy const&)
  {
    return *this;
  }
};

__host__ __device__ constexpr bool tests()
{
  {
    typedef cuda::std::array<double, 3> Array;
    Array array = {1.1, 2.2, 3.3};
    Array copy  = array;
    copy        = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    static_assert(cuda::std::is_copy_assignable<Array>::value, "");
    unused(copy);
  }
  {
    typedef cuda::std::array<double const, 3> Array;
    Array array = {1.1, 2.2, 3.3};
    Array copy  = array;
    unused(copy);
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    static_assert(!cuda::std::is_copy_assignable<Array>::value, "");
    unused(copy);
  }
  {
    typedef cuda::std::array<double, 0> Array;
    Array array = {};
    Array copy  = array;
    copy        = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    static_assert(cuda::std::is_copy_assignable<Array>::value, "");
    unused(copy);
  }
  {
    // const arrays of size 0 should disable the implicit copy assignment operator.
    typedef cuda::std::array<double const, 0> Array;
    Array array = {};
    Array copy  = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    static_assert(!cuda::std::is_copy_assignable<Array>::value, "");
    unused(copy);
  }
  {
    typedef cuda::std::array<NoDefault, 0> Array;
    Array array = {};
    Array copy  = array;
    copy        = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    static_assert(cuda::std::is_copy_assignable<Array>::value, "");
    unused(copy);
  }
  {
    typedef cuda::std::array<NoDefault const, 0> Array;
    Array array = {};
    Array copy  = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    static_assert(!cuda::std::is_copy_assignable<Array>::value, "");
    unused(copy);
  }

  // Make sure we can implicitly copy a cuda::std::array of a non-trivially copyable type
  {
    typedef cuda::std::array<NonTrivialCopy, 0> Array;
    Array array = {};
    Array copy  = array;
    copy        = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    unused(copy);
  }

// NVCC believes `copy = array` accesses uninitialized memory
#if TEST_CUDA_COMPILER(NVCC) || TEST_COMPILER(NVRTC)
  if (!TEST_IS_CONSTANT_EVALUATED())
#endif // TEST_CUDA_COMPILER(NVCC)
  {
    typedef cuda::std::array<NonTrivialCopy, 1> Array;
    Array array = {};
    Array copy  = array;
    copy        = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    unused(copy);
  }
// NVCC believes `copy = array` accesses uninitialized memory
#if TEST_CUDA_COMPILER(NVCC) || TEST_COMPILER(NVRTC)
  if (!TEST_IS_CONSTANT_EVALUATED())
#endif // TEST_CUDA_COMPILER(NVCC)
  {
    typedef cuda::std::array<NonTrivialCopy, 2> Array;
    Array array = {};
    Array copy  = array;
    copy        = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    unused(copy);
  }

  return true;
}

int main(int, char**)
{
  tests();
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  static_assert(tests(), "");
#endif
  return 0;
}
