/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 26, 2022.
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

// This test should pass in C++03 with Clang extensions because Clang does
// not implicitly delete the copy constructor when move constructors are
// defaulted using extensions.

// XFAIL: c++03

// test move

#include <uscl/std/cassert>
#include <uscl/std/utility>

struct move_only
{
  __host__ __device__ move_only() {}
  move_only(move_only&&)            = default;
  move_only& operator=(move_only&&) = default;
};

__host__ __device__ move_only source()
{
  return move_only();
}
__host__ __device__ const move_only csource()
{
  return move_only();
}

__host__ __device__ void test(move_only) {}

int main(int, char**)
{
  const move_only ca = move_only();
  // expected-error@+1 {{call to implicitly-deleted copy constructor of 'move_only'}}
  test(std::move(ca));

  return 0;
}
