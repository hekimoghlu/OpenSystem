/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <uscl/std/cassert>
#include <uscl/std/optional>

#include "test_macros.h"

template <class Value>
__host__ __device__ void
test(cuda::std::optional<Value> with_value, cuda::std::optional<Value> no_value, size_t expected_size)
{
  assert(with_value.value() == 42);
  assert(!no_value.has_value());
  assert(sizeof(cuda::std::optional<Value>) == expected_size);
}

template <class Value>
__global__ void
test_kernel(cuda::std::optional<Value> with_value, cuda::std::optional<Value> no_value, size_t expected_size)
{
  test(with_value, no_value, expected_size);
}

template <int Expected>
struct empty
{
  constexpr empty() = default;
  __host__ __device__ constexpr empty(const int val) noexcept
  {
    assert(val == Expected);
  }

  __host__ __device__ friend constexpr bool operator==(const empty&, int val)
  {
    return val == Expected;
  }
};

void test()
{
  { // non-empty payload
    using optional = cuda::std::optional<int>;
    optional with_value{cuda::std::in_place, 42};
    optional no_value{};
    test(with_value, no_value, sizeof(optional));
    test_kernel<<<1, 1>>>(with_value, no_value, sizeof(optional));
  }

  { // empty payload
    using optional = cuda::std::optional<empty<42>>;
    optional with_value{cuda::std::in_place, 42};
    optional no_value{};
    test(with_value, no_value, sizeof(optional));
    test_kernel<<<1, 1>>>(with_value, no_value, sizeof(optional));
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
