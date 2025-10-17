/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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
// UNSUPPORTED: pre-sm-70

#include <uscl/std/array>
#include <uscl/std/cassert>
#include <uscl/std/cstdint>
#include <uscl/std/type_traits>
#include <uscl/warp>

#include "test_macros.h"

template <typename T>
__device__ void test_types(T valueA = T{}, T valueB = T{1})
{
  for (int i = 1; i < 32; ++i)
  {
    auto mask = cuda::device::lane_mask{(1u << i) - 1};
    assert(cuda::device::warp_match_all(valueA, mask));
    if (i > 1)
    {
      [[maybe_unused]] auto value = threadIdx.x == 0 ? valueA : valueB;
      assert(!cuda::device::warp_match_all(value, mask));
    }
  }
}

__global__ void test_kernel()
{
  test_types<uint8_t>();
  test_types<uint16_t>();
  test_types<uint32_t>();
  test_types<uint64_t>();
#if _CCCL_HAS_INT128()
  test_types<__uint128_t>();
#endif
  test_types(char3{0, 0, 0}, char3{1, 1, 1});
  using array_t = cuda::std::array<char, 6>;
  test_types(array_t{0, 0, 0, 0, 0, 0}, array_t{1, 1, 1, 1, 1, 1});
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 32>>>();))
  return 0;
}
