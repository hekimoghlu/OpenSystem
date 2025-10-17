/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc

// uncomment for a really verbose output detailing what test steps are being launched
// #define DEBUG_TESTERS

#include <uscl/std/cassert>
#include <uscl/std/variant>

#include "helpers.h"

struct pod
{
  int val;

  __host__ __device__ friend bool operator==(pod lhs, pod rhs)
  {
    return lhs.val == rhs.val;
  }
};

using variant_t = cuda::std::variant<int, pod, double>;

template <typename T, int Val>
struct tester
{
  template <typename Variant>
  __host__ __device__ static void initialize(Variant&& v)
  {
    v = T{Val};
  }

  template <typename Variant>
  __host__ __device__ static void validate(Variant&& v)
  {
    assert(cuda::std::holds_alternative<T>(v));
    assert(cuda::std::get<T>(v) == T{Val});
  }
};

using testers =
  tester_list<tester<int, 10>, tester<int, 20>, tester<pod, 30>, tester<pod, 40>, tester<double, 50>, tester<double, 60>>;

void kernel_invoker()
{
  variant_t v;
  validate_pinned<variant_t, testers>(v);
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (kernel_invoker();))

  return 0;
}
