/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/atomic>

// cuda::atomic<key>

// Original test issue:
// https://github.com/NVIDIA/libcudacxx/issues/160

#include <uscl/atomic>

#include "cuda_space_selector.h"
#include "test_macros.h"

template <template <typename, typename> class Selector>
struct TestFn
{
  __host__ __device__ void operator()() const
  {
    {
      struct key
      {
        int32_t a;
        int32_t b;
      };
      typedef cuda::std::atomic<key> A;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      cuda::std::atomic_init(&t, key{1, 2});
      auto r = t.load();
      auto d = key{5, 5};
      t.store(r);
      (void) t.exchange(r);
      (void) t.compare_exchange_weak(r, d, cuda::memory_order_seq_cst, cuda::memory_order_seq_cst);
      (void) t.compare_exchange_strong(d, r, cuda::memory_order_seq_cst, cuda::memory_order_seq_cst);
    }
    {
      struct alignas(8) key
      {
        int32_t a;
        int32_t b;
      };
      typedef cuda::std::atomic<key> A;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      cuda::std::atomic_init(&t, key{1, 2});
      auto r = t.load();
      auto d = key{5, 5};
      t.store(r);
      (void) t.exchange(r);
      (void) t.compare_exchange_weak(r, d, cuda::memory_order_seq_cst, cuda::memory_order_seq_cst);
      (void) t.compare_exchange_strong(d, r, cuda::memory_order_seq_cst, cuda::memory_order_seq_cst);
    }
  }
};

int main(int, char**)
{
  NV_DISPATCH_TARGET(NV_IS_HOST, TestFn<local_memory_selector>()();
                     , NV_PROVIDES_SM_70, TestFn<local_memory_selector>()();)

  NV_IF_TARGET(NV_IS_DEVICE, (TestFn<shared_memory_selector>()(); TestFn<global_memory_selector>()();))

  return 0;
}
