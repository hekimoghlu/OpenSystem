/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 1, 2024.
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

// int memcmp(const void* lhs, const void* rhs, size_t count);

#include <uscl/std/cassert>
#include <uscl/std/cstddef>
#include <uscl/std/cstring>

#include "test_macros.h"

__host__ __device__ void test_memcmp(const char* lhs, const char* rhs, size_t n, int expected)
{
  const auto ret = cuda::std::memcmp(lhs, rhs, n);

  if (expected == 0)
  {
    assert(ret == 0);
  }
  else if (expected < 0)
  {
    assert(ret < 0);
  }
  else
  {
    assert(ret > 0);
  }
}

__host__ __device__ bool test()
{
  test_memcmp("abcde", "abcde", 5, 0);
  test_memcmp("abcd1", "abcd0", 5, 1);
  test_memcmp("abcd0", "abcd1", 5, -1);

  test_memcmp("abcd1", "abcd0", 4, 0);
  test_memcmp("abcd0", "abcd1", 4, 0);

  test_memcmp("abcde", "fghij", 5, -1);
  test_memcmp("abcde", "fghij", 0, 0);

  test_memcmp(nullptr, nullptr, 0, 0);

  return true;
}

int main(int, char**)
{
  test();
  return 0;
}
