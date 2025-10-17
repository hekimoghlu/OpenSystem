/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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

#include "concurrent_agents.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

template <template <typename, typename> class Selector>
__host__ __device__ void test()
{
  SHARED cuda::std::atomic_flag* t;
  execute_on_main_thread([&] {
    t = new cuda::std::atomic_flag();
    cuda::std::atomic_flag_clear(t);
    cuda::std::atomic_flag_wait(t, true);
  });

  auto agent_notify = LAMBDA()
  {
    assert(cuda::std::atomic_flag_test_and_set(t) == false);
    cuda::std::atomic_flag_notify_one(t);
  };

  auto agent_wait = LAMBDA()
  {
    cuda::std::atomic_flag_wait(t, false);
  };

  concurrent_agents_launch(agent_notify, agent_wait);

  SHARED volatile cuda::std::atomic_flag* vt;
  execute_on_main_thread([&] {
    vt = new cuda::std::atomic_flag();
    cuda::std::atomic_flag_clear(vt);
    cuda::std::atomic_flag_wait(vt, true);
  });

  auto agent_notify_v = LAMBDA()
  {
    assert(cuda::std::atomic_flag_test_and_set(vt) == false);
    cuda::std::atomic_flag_notify_one(vt);
  };

  auto agent_wait_v = LAMBDA()
  {
    cuda::std::atomic_flag_wait(vt, false);
  };

  concurrent_agents_launch(agent_notify_v, agent_wait_v);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, cuda_thread_count = 2;)

  test<shared_memory_selector>();

  return 0;
}
