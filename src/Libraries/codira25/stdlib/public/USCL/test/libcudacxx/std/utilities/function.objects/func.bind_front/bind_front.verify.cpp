/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// functional

// template <class F, class... Args>
// constexpr unspecified bind_front(F&&, Args&&...);

#include <uscl/std/functional>

#include "test_macros.h"

__host__ __device__ constexpr int pass(const int n)
{
  return n;
}

__host__ __device__ int simple(int n)
{
  return n;
}

template <class T>
__host__ __device__ T do_nothing(T t)
{
  return t;
}

struct NotMoveConst
{
  NotMoveConst(NotMoveConst&&)      = delete;
  NotMoveConst(NotMoveConst const&) = delete;

  __host__ __device__ NotMoveConst(int) {}
};

__host__ __device__ void testNotMoveConst(NotMoveConst) {}

int main(int, char**)
{
  int n       = 1;
  const int c = 1;

  auto p = cuda::std::bind_front(pass, c);
  static_assert(p() == 1); // expected-error-re {{{{(static_assert|static assertion)}} expression is not an integral
                           // constant expression}}

  auto d = cuda::std::bind_front(do_nothing, n); // expected-error {{no matching function for call to 'bind_front'}}

  auto t = cuda::std::bind_front(testNotMoveConst, NotMoveConst(0)); // expected-error {{no matching function for call
                                                                     // to 'bind_front'}}

  return 0;
}
