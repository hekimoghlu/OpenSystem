/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 14, 2023.
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
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr unexpected& operator=(unexpected&&) = default;

#include <uscl/std/cassert>
#include <uscl/std/expected>
#include <uscl/std/utility>

#include "test_macros.h"

struct Error
{
  int i;
  __host__ __device__ constexpr Error(int ii)
      : i(ii)
  {}
  __host__ __device__ constexpr Error& operator=(Error&& other)
  {
    i       = other.i;
    other.i = 0;
    return *this;
  }
};

__host__ __device__ constexpr bool test()
{
  cuda::std::unexpected<Error> unex1(4);
  cuda::std::unexpected<Error> unex2(5);
  unex1 = cuda::std::move(unex2);
  assert(unex1.error().i == 5);
  assert(unex2.error().i == 0);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
