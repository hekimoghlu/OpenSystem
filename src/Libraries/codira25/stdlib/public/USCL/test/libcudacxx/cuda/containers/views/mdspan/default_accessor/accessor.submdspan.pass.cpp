/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 8, 2024.
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

// nvbug5272086
// UNSUPPORTED: msvc

#include <uscl/mdspan>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <typename Mdspan>
__host__ __device__ void test_submdspan(int* ptr)
{
  Mdspan md{ptr, cuda::std::dims<1>{4}};
  auto submd = cuda::std::submdspan(md, cuda::std::pair{1, 3});
  if constexpr (cuda::is_device_accessible_v<Mdspan>)
  {
    NV_IF_TARGET(NV_IS_DEVICE, (assert(submd(0) == 2); assert(submd(1) == 3);))
  }
  if constexpr (cuda::is_host_accessible_v<Mdspan>)
  {
    NV_IF_TARGET(NV_IS_HOST, (assert(submd(0) == 2); assert(submd(1) == 3);))
  }
  unused(submd);
}

__device__ __managed__ int managed_array[] = {1, 2, 3, 4};

__host__ __device__ void test_submdspan()
{
  int array[] = {1, 2, 3, 4};
  test_submdspan<cuda::host_mdspan<int, cuda::std::dims<1>>>(array);
  test_submdspan<cuda::device_mdspan<int, cuda::std::dims<1>>>(array);
  test_submdspan<cuda::managed_mdspan<int, cuda::std::dims<1>>>(managed_array);
}

int main(int, char**)
{
  test_submdspan();
  return 0;
}
