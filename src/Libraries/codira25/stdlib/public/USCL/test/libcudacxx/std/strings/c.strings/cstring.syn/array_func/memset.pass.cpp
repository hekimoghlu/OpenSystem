/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 16, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// void* memset(void* s, int c, size_t n);

#include <uscl/std/cassert>
#include <uscl/std/cstring>

#include "test_macros.h"

template <typename T>
__host__ __device__ void test(int c)
{
  T obj{};
  assert(cuda::std::memset(&obj, c, sizeof(T)) == &obj);
  assert(cuda::std::memcmp(&obj, &obj, sizeof(T)) == 0);
}

struct SmallObj
{
  int i;
};

struct MidObj
{
  char i;
  int j;
  int k;
};

struct LargeObj
{
  short j;
  double ds[10];
};

int main(int, char**)
{
  test<char>(0);
  test<short>(255);
  test<int>(78);
  test<long>(127);
  test<float>(129);
  test<double>(200);
  test<SmallObj>(25);
  test<MidObj>(100);
  test<LargeObj>(187);
  test<void*>(0);
  test<int[10]>(2);

  return 0;
}
