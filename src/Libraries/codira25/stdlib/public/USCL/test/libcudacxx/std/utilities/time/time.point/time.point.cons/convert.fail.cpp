/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 2, 2023.
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

// time_point

// template <class Duration2>
//   time_point(const time_point<clock, Duration2>& t);

// Duration2 shall be implicitly convertible to duration.

// .fail. expects compilation to fail, but this would only fail at runtime with NVRTC

#include <uscl/std/chrono>

int main(int, char**)
{
  typedef cuda::std::chrono::system_clock Clock;
  typedef cuda::std::chrono::milliseconds Duration1;
  typedef cuda::std::chrono::microseconds Duration2;
  {
    cuda::std::chrono::time_point<Clock, Duration2> t2(Duration2(3));
    cuda::std::chrono::time_point<Clock, Duration1> t1 = t2;
  }

  return 0;
}
