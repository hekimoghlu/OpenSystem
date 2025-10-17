/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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

// UNSUPPORTED: pre-sm-80

#include <uscl/barrier>

#include "cuda_space_selector.h"
#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)
TEST_NV_DIAG_SUPPRESS(set_but_not_used)

__device__ void test()
{
  __shared__ cuda::barrier<cuda::thread_scope_block>* b;
  shared_memory_selector<cuda::barrier<cuda::thread_scope_block>, constructor_initializer> sel;
  b = sel.construct(2);

  [[maybe_unused]] uint64_t token;
  asm volatile("mbarrier.arrive.b64 %0, [%1];" : "=l"(token) : "l"(cuda::device::barrier_native_handle(*b)) : "memory");

  b->arrive_and_wait();
}

int main(int argc, char** argv)
{
  NV_IF_TARGET(NV_PROVIDES_SM_80, test();)

  return 0;
}
