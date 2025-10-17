/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 16, 2022.
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

// Make sure cuda::std::array is an aggregate type.
// We can only check this in C++17 and above, because we don't have the
// trait before that.

// UNSUPPORTED: gcc-6

#include <uscl/std/array>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <typename T>
__host__ __device__ void check_aggregate()
{
  static_assert(cuda::std::is_aggregate<cuda::std::array<T, 0>>::value, "");
  static_assert(cuda::std::is_aggregate<cuda::std::array<T, 1>>::value, "");
  static_assert(cuda::std::is_aggregate<cuda::std::array<T, 2>>::value, "");
  static_assert(cuda::std::is_aggregate<cuda::std::array<T, 3>>::value, "");
  static_assert(cuda::std::is_aggregate<cuda::std::array<T, 4>>::value, "");
}

struct Empty
{};
struct Trivial
{
  int i;
  int j;
};
struct NonTrivial
{
  int i;
  int j;
  __host__ __device__ NonTrivial(NonTrivial const&) {}
};

int main(int, char**)
{
  check_aggregate<char>();
  check_aggregate<int>();
  check_aggregate<long>();
  check_aggregate<float>();
  check_aggregate<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  check_aggregate<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
  check_aggregate<Empty>();
  check_aggregate<Trivial>();
  check_aggregate<NonTrivial>();

  return 0;
}
