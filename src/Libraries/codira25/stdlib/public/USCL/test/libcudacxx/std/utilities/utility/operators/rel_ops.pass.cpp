/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 28, 2023.
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

// test rel_ops

#include <uscl/std/cassert>
#include <uscl/std/utility>

#include "test_macros.h"

struct A
{
  int data_;

  __host__ __device__ explicit A(int data = -1)
      : data_(data)
  {}
};

inline __host__ __device__ bool operator==(const A& x, const A& y)
{
  return x.data_ == y.data_;
}

inline __host__ __device__ bool operator<(const A& x, const A& y)
{
  return x.data_ < y.data_;
}

int main(int, char**)
{
  using namespace cuda::std::rel_ops;
  A a1(1);
  A a2(2);
  assert(a1 == a1);
  assert(a1 != a2);
  assert(a1 < a2);
  assert(a2 > a1);
  assert(a1 <= a1);
  assert(a1 <= a2);
  assert(a2 >= a2);
  assert(a2 >= a1);

  return 0;
}
