/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 25, 2023.
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

// <cuda/std/chrono>

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

int main(int, char**)
{
  using namespace cuda::std::literals;

  cuda::std::chrono::hours h = 4h;
  assert(h == cuda::std::chrono::hours(4));
  auto h2 = 4.0h;
  assert(h == h2);

  cuda::std::chrono::minutes min = 36min;
  assert(min == cuda::std::chrono::minutes(36));
  auto min2 = 36.0min;
  assert(min == min2);

  cuda::std::chrono::seconds s = 24s;
  assert(s == cuda::std::chrono::seconds(24));
  auto s2 = 24.0s;
  assert(s == s2);

  cuda::std::chrono::milliseconds ms = 247ms;
  assert(ms == cuda::std::chrono::milliseconds(247));
  auto ms2 = 247.0ms;
  assert(ms == ms2);

  cuda::std::chrono::microseconds us = 867us;
  assert(us == cuda::std::chrono::microseconds(867));
  auto us2 = 867.0us;
  assert(us == us2);

  cuda::std::chrono::nanoseconds ns = 645ns;
  assert(ns == cuda::std::chrono::nanoseconds(645));
  auto ns2 = 645.ns;
  assert(ns == ns2);

  return 0;
}
