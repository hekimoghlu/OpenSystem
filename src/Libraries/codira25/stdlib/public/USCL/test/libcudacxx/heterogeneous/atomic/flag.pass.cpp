/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

#include <uscl/std/atomic>
#include <uscl/std/cassert>

#include "../helpers.h"

struct clear
{
  template <typename AF>
  __host__ __device__ static void initialize(AF& af)
  {
    af.clear();
  }
};

struct clear_tester : clear
{
  template <typename AF>
  __host__ __device__ static void validate(AF& af)
  {
    assert(af.test_and_set() == false);
  }
};

template <bool Previous>
struct test_and_set_tester
{
  template <typename AF>
  __host__ __device__ static void initialize(AF& af)
  {
    assert(af.test_and_set() == Previous);
  }

  template <typename AF>
  __host__ __device__ static void validate(AF& af)
  {
    assert(af.test_and_set() == true);
  }
};

using atomic_flag_testers = tester_list<clear_tester, clear, test_and_set_tester<false>, test_and_set_tester<true>>;

void kernel_invoker()
{
  validate_pinned<cuda::std::atomic_flag, atomic_flag_testers>();
}

int main(int argc, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (kernel_invoker();))

  return 0;
}
