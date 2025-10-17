/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 25, 2024.
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
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: pre-sm-70

// <cuda/std/atomic>

#include <uscl/std/atomic>
#include <uscl/std/cassert>
#include <uscl/std/type_traits>

#include "../atomics.types.operations.req/atomic_helpers.h"
#include "concurrent_agents.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

template <class T, template <typename, typename> class Selector, cuda::thread_scope Scope>
struct TestFn
{
  __host__ __device__ void operator()() const
  {
    typedef cuda::std::atomic_ref<T> A;

    SHARED T* t;
    execute_on_main_thread([&] {
      t = (T*) malloc(sizeof(A));
      A a(*t);
      a.store(T(1));
      assert(a.load() == T(1));
      a.wait(T(0));
    });

    {
      A a(*t);

      auto agent_notify = LAMBDA()
      {
        a.store(T(3));
        a.notify_one();
      };

      auto agent_wait = LAMBDA()
      {
        a.wait(T(1));
      };

      concurrent_agents_launch(agent_notify, agent_wait);
    }
  }
};

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, cuda_thread_count = 2;)

  TestEachAtomicRefType<TestFn, shared_memory_selector>()();

  return 0;
}
