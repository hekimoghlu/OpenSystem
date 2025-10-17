/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 4, 2021.
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

#include "utils.h"
#define ARR_SZ 128

template <typename T, typename P>
__device__ __host__ __noinline__ void test(P ap)
{
  T* arr = global_alloc<T, ARR_SZ>();

  arr = cuda::associate_access_property(arr, ap);

  for (int i = 0; i < ARR_SZ; ++i)
  {
    assert(arr[i] == i);
  }

  dealloc<T>(arr);
}

template <typename T, typename P>
__device__ __host__ __noinline__ void test_shared(P ap)
{
  T* arr = shared_alloc<T, ARR_SZ>();

  arr = cuda::associate_access_property(arr, ap);

  for (int i = 0; i < ARR_SZ; ++i)
  {
    assert(arr[i] == i);
  }
}

__device__ __host__ __noinline__ void test_all()
{
  test<int>(cuda::access_property::normal{});
  test<int>(cuda::access_property::persisting{});
  test<int>(cuda::access_property::streaming{});
  test<int>(cuda::access_property::global{});
  test<int>(cuda::access_property{});
  NV_IF_TARGET(NV_IS_DEVICE, (test_shared<int>(cuda::access_property::shared{});))
}

int main(int argc, char** argv)
{
  test_all();
  return 0;
}
