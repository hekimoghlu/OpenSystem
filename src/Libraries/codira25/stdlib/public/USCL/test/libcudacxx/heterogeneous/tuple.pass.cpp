/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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

// UNSUPPORTED: nvrtc

// uncomment for a really verbose output detailing what test steps are being launched
// #define DEBUG_TESTERS

#include <uscl/std/cassert>
#include <uscl/std/tuple>

#include "helpers.h"

struct pod
{
  char val[10];
};

using tuple_t = cuda::std::tuple<int, pod, unsigned long long>;

template <int N>
struct Write
{
  using async = cuda::std::false_type;

  template <typename Tuple>
  __host__ __device__ static void perform(Tuple& t)
  {
    cuda::std::get<0>(t)        = N;
    cuda::std::get<1>(t).val[0] = N;
    cuda::std::get<2>(t)        = N;
  }
};

template <int N>
struct Read
{
  using async = cuda::std::false_type;

  template <typename Tuple>
  __host__ __device__ static void perform(Tuple& t)
  {
    assert(cuda::std::get<0>(t) == N);
    assert(cuda::std::get<1>(t).val[0] == N);
    assert(cuda::std::get<2>(t) == N);
  }
};

using w_r_w_r = performer_list<Write<10>, Read<10>, Write<30>, Read<30>>;

void kernel_invoker()
{
  tuple_t t(0, {0}, 0);
  validate_pinned<tuple_t, w_r_w_r>(t);
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (kernel_invoker();))

  return 0;
}
