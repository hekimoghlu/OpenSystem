/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
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
// UNSUPPORTED: c++17

#include <uscl/std/atomic>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(cuda::std::memory_order_relaxed == cuda::std::memory_order::relaxed);
  static_assert(cuda::std::memory_order_consume == cuda::std::memory_order::consume);
  static_assert(cuda::std::memory_order_acquire == cuda::std::memory_order::acquire);
  static_assert(cuda::std::memory_order_release == cuda::std::memory_order::release);
  static_assert(cuda::std::memory_order_acq_rel == cuda::std::memory_order::acq_rel);
  static_assert(cuda::std::memory_order_seq_cst == cuda::std::memory_order::seq_cst);

  return 0;
}
