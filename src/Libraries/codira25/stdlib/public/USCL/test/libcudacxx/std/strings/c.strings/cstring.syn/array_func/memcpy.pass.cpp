/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 23, 2022.
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

// void* memcpy(void* dst, const void* src, size_t count);

#include <uscl/std/cassert>
#include <uscl/std/cstring>

#include "test_macros.h"

template <typename T>
__host__ __device__ void test(T obj)
{
  unsigned char buf[sizeof(T)]{};
  assert(cuda::std::memcpy(buf, &obj, sizeof(T)) == buf);
  assert(cuda::std::memcmp(buf, &obj, sizeof(T)) == 0);
}

struct SmallObj
{
  int i;
};

struct MidObj
{
  char c1;
  char c2;
  short s;
  int j;
  int k;
};

struct LargeObj
{
  double ds[10];
};

union Union
{
  int i;
  double d;
};

int main(int, char**)
{
  test('a');
  test(short(2489));
  test(780581);
  test(127156178992l);
  test(129.f);
  test(20123.003445);
  test(SmallObj{25});
  test(MidObj{'a', 'b', 120, 902183, 3124});
  test(LargeObj{187.0, 0.00000346, 1203980985.4365, 123.567, 0.0, -0.0, 123.567});
  test(reinterpret_cast<char*>(123456));
  test(Union{123456});

  return 0;
}
