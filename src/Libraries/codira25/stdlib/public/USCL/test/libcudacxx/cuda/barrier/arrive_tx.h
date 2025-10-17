/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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
#ifndef TEST_ARRIVE_TX_H_
#define TEST_ARRIVE_TX_H_

#include <uscl/barrier>
#include <uscl/memory>
#include <uscl/std/utility>

#include "concurrent_agents.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

// Suppress warning about barrier in shared memory
TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)

template <typename Barrier>
inline __device__ void mbarrier_complete_tx(Barrier& b, int transaction_count)
{
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_90,
    (
      if (cuda::device::is_address_from(cuda::device::barrier_native_handle(b), cuda::device::address_space::shared)) {
        asm volatile(
          "mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
          :
          : "r"((unsigned int) __cvta_generic_to_shared(cuda::device::barrier_native_handle(b))), "r"(transaction_count)
          : "memory");
      } else { __trap(); }),
    NV_ANY_TARGET,
    (
      // On architectures pre-SM90 (and on host), we drop the transaction count
      // update. The barriers do not keep track of transaction counts.
      __trap();));
}

template <bool split_arrive_and_expect>
__device__ void thread(cuda::barrier<cuda::thread_scope_block>& b, int arrives_per_thread)
{
  constexpr int tx_count = 1;
  typename cuda::barrier<cuda::thread_scope_block>::arrival_token tok;

  if _CCCL_CONSTEXPR_CXX20 (split_arrive_and_expect)
  {
    cuda::device::barrier_expect_tx(b, tx_count);
    tok = b.arrive(arrives_per_thread);
  }
  else
  {
    tok = cuda::device::barrier_arrive_tx(b, arrives_per_thread, tx_count);
  }

  // Manually increase the transaction count of the barrier.
  mbarrier_complete_tx(b, tx_count);

  b.wait(cuda::std::move(tok));
}

template <bool split_arrive_and_expect>
__device__ void test()
{
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE,
    (
      // Run all threads, each arriving with arrival count 1
      using barrier_t = cuda::barrier<cuda::thread_scope_block>;

      shared_memory_selector<barrier_t, constructor_initializer> sel_1;
      barrier_t* bar_1 = sel_1.construct(blockDim.x);
      __syncthreads();
      thread<split_arrive_and_expect>(*bar_1, 1);

      // Run all threads, each arriving with arrival count 2
      shared_memory_selector<barrier_t, constructor_initializer> sel_2;
      barrier_t* bar_2 = sel_2.construct(2 * blockDim.x);
      __syncthreads();
      thread<split_arrive_and_expect>(*bar_2, 2);));
}

#endif // TEST_ARRIVE_TX_H_
