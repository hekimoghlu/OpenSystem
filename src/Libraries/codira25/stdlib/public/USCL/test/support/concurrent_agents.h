/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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

#ifndef _CONCURRENT_AGENTS_H
#define _CONCURRENT_AGENTS_H

#ifndef __CUDA_ARCH__
#  include <thread>
#endif

#include <uscl/std/cassert>

#include "test_macros.h"

_CCCL_EXEC_CHECK_DISABLE
template <class Fun>
__host__ __device__ void execute_on_main_thread(Fun&& fun)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (if (threadIdx.x == 0) { fun(); } __syncthreads();), (fun();))
}

template <typename... Fs>
__host__ __device__ void concurrent_agents_launch(Fs... fs)
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (assert(blockDim.x == sizeof...(Fs)); using fptr = void (*)(void*);

     fptr device_threads[] = {[](void* data) {
       (*reinterpret_cast<Fs*>(data))();
     }...};

     void* device_thread_data[] = {reinterpret_cast<void*>(&fs)...};

     __syncthreads();

     device_threads[threadIdx.x](device_thread_data[threadIdx.x]);

     __syncthreads();),
    (std::thread threads[]{std::thread{std::forward<Fs>(fs)}...};

     for (auto&& thread : threads) { thread.join(); }))
}

#endif // _CONCURRENT_AGENTS_H
