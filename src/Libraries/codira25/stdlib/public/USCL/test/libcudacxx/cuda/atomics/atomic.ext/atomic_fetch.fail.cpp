/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 21, 2022.
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

#include <uscl/atomic>
#include <uscl/std/cassert>
#include <uscl/std/type_traits>

#include "atomic_helpers.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

template <class T, template <typename, typename> class Selector, cuda::thread_scope>
struct TestFn
{
  __host__ __device__ void operator()() const
  {
    {
      typedef cuda::atomic<T> A;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      t.fetch_min(4);
    }
    {
      typedef cuda::atomic<T> A;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      t.fetch_max(4);
    }
    T tmp = T(0);
    {
      cuda::atomic_ref<T> t(tmp);
      t.fetch_min(4);
    }
    {
      cuda::atomic_ref<T> t(tmp);
      t.fetch_max(4);
    }
  }
};

int main(int, char**)
{
  NV_DISPATCH_TARGET(
    NV_IS_HOST, TestFn<__half, local_memory_selector, cuda::thread_scope::thread_scope_thread>()();
    , NV_PROVIDES_SM_70, (TestFn<__half, local_memory_selector, cuda::thread_scope::thread_scope_thread>()();))

  NV_IF_TARGET(NV_IS_DEVICE,
               (TestFn<__half, shared_memory_selector, cuda::thread_scope::thread_scope_thread>()();
                TestFn<__half, global_memory_selector, cuda::thread_scope::thread_scope_thread>()();))

  return 0;
}
