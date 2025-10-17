/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 5, 2025.
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

#include <uscl/memory>
#include <uscl/std/cassert>

#include "test_macros.h"

__device__ int scalar_object             = 42;
__device__ const int const_scalar_object = 42;

__device__ int array_object[]             = {42, 1337, -1};
__device__ const int const_array_object[] = {42, 1337, -1};

#if !TEST_COMPILER(NVRTC)
template <class T>
void test_host(T& object)
{
  {
    T* host_address = cuda::std::addressof(object);

    cudaPointerAttributes attributes;
    cudaError_t status = cudaPointerGetAttributes(&attributes, host_address);
    assert(status == cudaSuccess);

    if (attributes.devicePointer)
    {
      assert(attributes.devicePointer == host_address);
    }
  }

  {
    T* device_address = cuda::get_device_address(object);

    cudaPointerAttributes attributes;
    cudaError_t status = cudaPointerGetAttributes(&attributes, device_address);
    assert(status == cudaSuccess);
    assert(attributes.devicePointer == device_address);
  }
}
#endif // !TEST_COMPILER(NVRTC)

template <class T>
__host__ __device__ void test(T& object)
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (assert(cuda::std::addressof(object) == cuda::get_device_address(object));), (test_host(object);))
}

int main(int argc, char** argv)
{
  test(scalar_object);
  test(const_scalar_object);
  test(array_object);
  test(const_array_object);

  return 0;
}
