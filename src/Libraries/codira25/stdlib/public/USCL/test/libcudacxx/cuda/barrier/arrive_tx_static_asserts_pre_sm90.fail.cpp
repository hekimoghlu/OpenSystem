/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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

// UNSUPPORTED: pre-sm-70

// <cuda/barrier>

#include <uscl/barrier>

int main(int, char**)
{
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (__shared__ cuda::barrier<cuda::thread_scope_block> bar;
     if (threadIdx.x == 0) { init(&bar, blockDim.x); } __syncthreads();

     // barrier_arrive_tx should fail on SM70 and SM80, because it is hidden.
     auto token = cuda::device::barrier_arrive_tx(bar, 1, 0);

#ifdef __cccl_lib_local_barrier_arrive_tx
     static_assert(false, "Fail manually for SM90 and up.");
#endif // __cccl_lib_local_barrier_arrive_tx
     ));
  return 0;
}
