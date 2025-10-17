/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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
#include <uscl/std/array>
#include <uscl/std/cassert>
#include <uscl/std/cstdint>
#include <uscl/std/type_traits>
#include <uscl/warp>

#include "test_macros.h"

template <int Value>
inline constexpr auto width_v = cuda::std::integral_constant<int, Value>{};

template <int Value>
__device__ void test_semantic()
{
  using cuda::device::warp_shuffle_down;
  using cuda::device::warp_shuffle_idx;
  using cuda::device::warp_shuffle_up;
  using cuda::device::warp_shuffle_xor;
  uint32_t data           = threadIdx.x;
  constexpr uint32_t mask = 0xFFFFFFFF;
  for (int i = 0; i < 64; i++)
  {
    assert(warp_shuffle_idx(data, i, mask, width_v<Value>) == __shfl_sync(mask, data, i, Value));
  }
  for (int i = 1; i < Value; i++)
  {
    assert(warp_shuffle_down(data, i, mask, width_v<Value>) == __shfl_down_sync(mask, data, i, Value));
    assert(warp_shuffle_up(data, i, mask, width_v<Value>) == __shfl_up_sync(mask, data, i, Value));
    assert(warp_shuffle_xor(data, i, mask, width_v<Value>) == __shfl_xor_sync(mask, data, i, Value));
  }
  unused(data);
  unused(mask);
  if (Value == 16 && threadIdx.x < 16)
  {
    constexpr uint32_t mask2 = 0xFFFF;
    int i                    = 4;
    assert(warp_shuffle_idx(data, i, mask2, width_v<Value>) == __shfl_sync(mask2, data, i, Value));
    assert(warp_shuffle_down(data, i, mask2, width_v<Value>) == __shfl_down_sync(mask2, data, i, Value));
    assert(warp_shuffle_xor(data, i, mask2, width_v<Value>) == __shfl_xor_sync(mask2, data, i, Value));
    assert(warp_shuffle_up(data, i, mask2, width_v<Value>) == __shfl_up_sync(mask2, data, i, Value));
    assert(warp_shuffle_up<16>(data, i, mask2) == __shfl_up_sync(mask2, data, i, Value));
  }
}

template <class T>
__device__ void test_non_trivial_types(const T& data)
{
  using cuda::device::warp_shuffle_down;
  using cuda::device::warp_shuffle_idx;
  using cuda::device::warp_shuffle_up;
  using cuda::device::warp_shuffle_xor;
  const T default_value{};
  // idx
  {
    auto& data1 = threadIdx.x == 0 ? data : default_value;
    auto ret    = warp_shuffle_idx(data1, 0);
    assert(ret.data[0] == data[0] && ret.data[1] == data[1] && ret.data[2] == data[2] && ret.data[3] == data[3]);
    unused(ret);
  }
  {
    // down
    auto& data1 = threadIdx.x >= 2 ? data : default_value;
    auto ret    = warp_shuffle_down(data1, 2);
    assert(ret.data[0] == data[0] && ret.data[1] == data[1] && ret.data[2] == data[2] && ret.data[3] == data[3]);
    unused(ret);
  }
  {
    // up
    auto& data1 = threadIdx.x < 30 ? data : default_value;
    auto ret    = warp_shuffle_up(data1, 2);
    assert(ret.data[0] == data[0] && ret.data[1] == data[1] && ret.data[2] == data[2] && ret.data[3] == data[3]);
    unused(ret);
  }
  {
    // xor
    auto& data1   = threadIdx.x % 2 == 0 ? data : default_value;
    auto ret      = warp_shuffle_xor(data1, 1);
    auto cmp_data = threadIdx.x % 2 == 0 ? default_value : data;
    assert(ret.data[0] == cmp_data[0] && ret.data[1] == cmp_data[1] && ret.data[2] == cmp_data[2]
           && ret.data[3] == cmp_data[3]);
    unused(ret);
    unused(cmp_data);
  }
}

__device__ void test_overloadings()
{
  using cuda::device::warp_shuffle_down;
  using cuda::device::warp_shuffle_idx;
  using cuda::device::warp_shuffle_up;
  using cuda::device::warp_shuffle_xor;
  uint32_t data           = threadIdx.x;
  constexpr uint32_t mask = 0xFFFFFFFF;
  for (int i = 0; i < 64; i++)
  {
    assert(warp_shuffle_idx(data, i, mask) == __shfl_sync(mask, data, i));
  }
  for (int i = 1; i < 32; i++)
  {
    assert(warp_shuffle_down(data, i, mask) == __shfl_down_sync(mask, data, i));
    assert(warp_shuffle_up(data, i, mask) == __shfl_up_sync(mask, data, i));
    assert(warp_shuffle_xor(data, i, mask) == __shfl_xor_sync(mask, data, i));
  }
}

__global__ void test_kernel()
{
  test_semantic<1>();
  test_semantic<2>();
  test_semantic<4>();
  test_semantic<8>();
  test_semantic<16>();
  test_semantic<32>();
  test_overloadings();
  test_non_trivial_types(cuda::std::array<double, 4>{1.0, 2.0, 3.0, 4.0});
  double array[4] = {1.0, 2.0, 3.0, 4.0};
  test_non_trivial_types(array);
  auto ptr = threadIdx.x == 0 ? static_cast<const void*>(array) : nullptr;
  assert(cuda::device::warp_shuffle_idx(ptr, 0) == static_cast<const void*>(array));
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 32>>>();))
  return 0;
}
