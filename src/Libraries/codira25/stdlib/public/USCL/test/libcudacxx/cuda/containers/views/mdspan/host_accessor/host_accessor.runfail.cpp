/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 7, 2024.
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

#include <uscl/mdspan>

#include "test_macros.h"

__host__ void host_accessor_runtime_fail()
{
  int* device_ptr = nullptr;
  assert(cudaMalloc(&device_ptr, 4) == cudaSuccess);
  using ext_t = cuda::std::extents<int, 4>;
  cuda::host_mdspan<int, ext_t> h_md{device_ptr, ext_t{}};
  NV_IF_TARGET(NV_IS_HOST, (unused(h_md[0]);))
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, host_accessor_runtime_fail();)
  NV_IF_ELSE_TARGET(NV_IS_HOST, (return 0;), (return 1;))
}
