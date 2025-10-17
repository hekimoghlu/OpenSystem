/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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

// <cuda/std/semaphore>

#include <uscl/std/semaphore>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(cuda::std::counting_semaphore<>::max() > 0, "");
  static_assert(cuda::std::counting_semaphore<1>::max() >= 1, "");
  static_assert(cuda::std::counting_semaphore<cuda::std::numeric_limits<int>::max()>::max() >= 1, "");
  static_assert(cuda::std::counting_semaphore<cuda::std::numeric_limits<unsigned>::max()>::max() >= 1, "");
  static_assert(cuda::std::counting_semaphore<cuda::std::numeric_limits<ptrdiff_t>::max()>::max() >= 1, "");
  static_assert(cuda::std::counting_semaphore<1>::max() == cuda::std::binary_semaphore::max(), "");

  static_assert(cuda::counting_semaphore<cuda::thread_scope_system>::max() > 0, "");
  static_assert(cuda::counting_semaphore<cuda::thread_scope_system, 1>::max() >= 1, "");
  static_assert(cuda::counting_semaphore<cuda::thread_scope_system, cuda::std::numeric_limits<int>::max()>::max() >= 1,
                "");
  static_assert(
    cuda::counting_semaphore<cuda::thread_scope_system, cuda::std::numeric_limits<unsigned>::max()>::max() >= 1, "");
  static_assert(
    cuda::counting_semaphore<cuda::thread_scope_system, cuda::std::numeric_limits<ptrdiff_t>::max()>::max() >= 1, "");
  static_assert(cuda::counting_semaphore<cuda::thread_scope_system, 1>::max()
                  == cuda::binary_semaphore<cuda::thread_scope_system>::max(),
                "");

  static_assert(cuda::counting_semaphore<cuda::thread_scope_device>::max() > 0, "");
  static_assert(cuda::counting_semaphore<cuda::thread_scope_device, 1>::max() >= 1, "");
  static_assert(cuda::counting_semaphore<cuda::thread_scope_device, cuda::std::numeric_limits<int>::max()>::max() >= 1,
                "");
  static_assert(
    cuda::counting_semaphore<cuda::thread_scope_device, cuda::std::numeric_limits<unsigned>::max()>::max() >= 1, "");
  static_assert(
    cuda::counting_semaphore<cuda::thread_scope_device, cuda::std::numeric_limits<ptrdiff_t>::max()>::max() >= 1, "");
  static_assert(cuda::counting_semaphore<cuda::thread_scope_device, 1>::max()
                  == cuda::binary_semaphore<cuda::thread_scope_device>::max(),
                "");

  static_assert(cuda::counting_semaphore<cuda::thread_scope_block>::max() > 0, "");
  static_assert(cuda::counting_semaphore<cuda::thread_scope_block, 1>::max() >= 1, "");
  static_assert(cuda::counting_semaphore<cuda::thread_scope_block, cuda::std::numeric_limits<int>::max()>::max() >= 1,
                "");
  static_assert(
    cuda::counting_semaphore<cuda::thread_scope_block, cuda::std::numeric_limits<unsigned>::max()>::max() >= 1, "");
  static_assert(
    cuda::counting_semaphore<cuda::thread_scope_block, cuda::std::numeric_limits<ptrdiff_t>::max()>::max() >= 1, "");
  static_assert(cuda::counting_semaphore<cuda::thread_scope_block, 1>::max()
                  == cuda::binary_semaphore<cuda::thread_scope_block>::max(),
                "");

  return 0;
}
