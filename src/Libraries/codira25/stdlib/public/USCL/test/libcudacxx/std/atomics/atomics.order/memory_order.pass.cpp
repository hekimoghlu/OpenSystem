/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 3, 2022.
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

// <cuda/std/atomic>

// typedef enum memory_order
// {
//     memory_order_relaxed, memory_order_consume, memory_order_acquire,
//     memory_order_release, memory_order_acq_rel, memory_order_seq_cst
// } memory_order;

#include <uscl/std/atomic>
#include <uscl/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
  assert(static_cast<int>(cuda::std::memory_order_relaxed) == 0);
  assert(static_cast<int>(cuda::std::memory_order_consume) == 1);
  assert(static_cast<int>(cuda::std::memory_order_acquire) == 2);
  assert(static_cast<int>(cuda::std::memory_order_release) == 3);
  assert(static_cast<int>(cuda::std::memory_order_acq_rel) == 4);
  assert(static_cast<int>(cuda::std::memory_order_seq_cst) == 5);

  cuda::std::memory_order o = cuda::std::memory_order_seq_cst;
  assert(static_cast<int>(o) == 5);

  return 0;
}
