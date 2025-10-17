/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc, pre-sm-70
// UNSUPPORTED: true

// uncomment for a really verbose output detailing what test steps are being launched
// #define DEBUG_TESTERS

#include <uscl/barrier>
#include <uscl/std/cassert>

#include <atomic>

#include "helpers.h"

template <typename Barrier>
struct barrier_and_token
{
  using barrier_t = Barrier;
  using token_t   = typename barrier_t::arrival_token;

  barrier_t barrier;
  cuda::std::atomic<bool> parity_waiting{false};

  template <typename... Args>
  __host__ __device__ barrier_and_token(Args&&... args)
      : barrier{cuda::std::forward<Args>(args)...}
  {}
};

struct barrier_arrive_and_wait
{
  using async = cuda::std::true_type;

  template <typename Data>
  __host__ __device__ static void perform(Data& data)
  {
    while (data.parity_waiting.load(cuda::std::memory_order_acquire) == false)
    {
      data.parity_waiting.wait(false);
    }
    data.barrier.arrive_and_wait();
  }
};

template <bool Phase>
struct barrier_parity_wait
{
  using async = cuda::std::true_type;

  template <typename Data>
  __host__ __device__ static void perform(Data& data)
  {
    data.parity_waiting.store(true, cuda::std::memory_order_release);
    data.parity_waiting.notify_all();
    data.barrier.wait_parity(Phase);
  }
};

struct clear_token
{
  template <typename Data>
  __host__ __device__ static void perform(Data& data)
  {
    data.parity_waiting.store(false, cuda::std::memory_order_release);
  }
};

using aw_aw_pw1 =
  performer_list<barrier_parity_wait<false>,
                 barrier_arrive_and_wait,
                 barrier_arrive_and_wait,
                 async_tester_fence,
                 clear_token>;

using aw_aw_pw2 =
  performer_list<barrier_parity_wait<true>,
                 barrier_arrive_and_wait,
                 barrier_arrive_and_wait,
                 async_tester_fence,
                 clear_token>;

void kernel_invoker()
{
  validate_pinned<barrier_and_token<cuda::barrier<cuda::thread_scope_system>>, aw_aw_pw1>(2);
  validate_pinned<barrier_and_token<cuda::barrier<cuda::thread_scope_system>>, aw_aw_pw2>(2);
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (kernel_invoker();))

  return 0;
}
