/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 14, 2022.
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
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: pre-sm-90

// <cuda/barrier>

#include <uscl/barrier>

#include "test_macros.h" // TEST_NV_DIAG_SUPPRESS

// Suppress warning about barrier in shared memory
TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// Kernels below are intended to be compiled, but not run. This is to check if
// all generated PTX is valid.
__global__ void test_bulk_tensor(CUtensorMap* map)
{
  __shared__ int smem;
#if _CCCL_CUDA_COMPILER(CLANG)
  __shared__ char barrier_data[sizeof(barrier)];
  barrier& bar = cuda::std::bit_cast<barrier>(barrier_data);
#else // ^^^ _CCCL_CUDA_COMPILER(CLANG) ^^^ / vvv !_CCCL_CUDA_COMPILER(CLANG)
  __shared__ barrier bar;
#endif // !_CCCL_CUDA_COMPILER(CLANG)
  if (threadIdx.x == 0)
  {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  cde::cp_async_bulk_tensor_1d_global_to_shared(&smem, map, 0, bar);
  cde::cp_async_bulk_tensor_2d_global_to_shared(&smem, map, 0, 0, bar);
  cde::cp_async_bulk_tensor_3d_global_to_shared(&smem, map, 0, 0, 0, bar);
  cde::cp_async_bulk_tensor_4d_global_to_shared(&smem, map, 0, 0, 0, 0, bar);
  cde::cp_async_bulk_tensor_5d_global_to_shared(&smem, map, 0, 0, 0, 0, 0, bar);

  cde::cp_async_bulk_tensor_1d_shared_to_global(map, 0, &smem);
  cde::cp_async_bulk_tensor_2d_shared_to_global(map, 0, 0, &smem);
  cde::cp_async_bulk_tensor_3d_shared_to_global(map, 0, 0, 0, &smem);
  cde::cp_async_bulk_tensor_4d_shared_to_global(map, 0, 0, 0, 0, &smem);
  cde::cp_async_bulk_tensor_5d_shared_to_global(map, 0, 0, 0, 0, 0, &smem);
}

__global__ void test_bulk(void* gmem)
{
  __shared__ int smem;
  __shared__ char barrier_data[sizeof(barrier)];
  barrier& bar = *reinterpret_cast<barrier*>(&barrier_data);
  if (threadIdx.x == 0)
  {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  cde::cp_async_bulk_global_to_shared(&smem, gmem, 1024, bar);
  cde::cp_async_bulk_shared_to_global(gmem, &smem, 1024);
}

__global__ void test_fences_async_group(void*)
{
  cde::fence_proxy_async_shared_cta();

  cde::cp_async_bulk_commit_group();
  // Wait for up to 8 groups
  cde::cp_async_bulk_wait_group_read<0>();
  cde::cp_async_bulk_wait_group_read<1>();
  cde::cp_async_bulk_wait_group_read<2>();
  cde::cp_async_bulk_wait_group_read<3>();
  cde::cp_async_bulk_wait_group_read<4>();
  cde::cp_async_bulk_wait_group_read<5>();
  cde::cp_async_bulk_wait_group_read<6>();
  cde::cp_async_bulk_wait_group_read<7>();
  cde::cp_async_bulk_wait_group_read<8>();
}

int main(int, char**)
{
  return 0;
}
