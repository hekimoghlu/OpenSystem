/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 3, 2025.
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

// void* memmove(void* dst, const void* src, size_t count);

#include <uscl/std/cassert>
#include <uscl/std/cstring>

#include "test_macros.h"

__host__ __device__ bool test_out_of_place()
{
  char src[] = "1234567890";

  {
    char dest[] = "abcdefghij";
    char ref[]  = "1234567890";
    assert(cuda::std::memmove(dest, src, 10) == dest);
    assert(cuda::std::memcmp(dest, ref, 10) == 0);
  }

  {
    char dest[] = "abcdefghij";
    char ref[]  = "abc123456j";
    assert(cuda::std::memmove(dest + 3, src, 6) == dest + 3);
    assert(cuda::std::memcmp(dest, ref, 10) == 0);
  }

  {
    char dest[] = "abcdefghij";
    char ref[]  = "56789fghij";
    assert(cuda::std::memmove(dest, src + 4, 5) == dest);
    assert(cuda::std::memcmp(dest, ref, 10) == 0);
  }

  {
    char dest[] = "abcdefghij";
    char ref[]  = "ab789fghij";
    assert(cuda::std::memmove(dest + 2, src + 6, 3) == dest + 2);
    assert(cuda::std::memcmp(dest, ref, 10) == 0);
  }

  return true;
}

__host__ __device__ bool test_in_place()
{
  {
    char buf[] = "1234567890";
    char ref[] = "1234567890";
    assert(cuda::std::memmove(buf, buf, 10) == buf);
    assert(cuda::std::memcmp(buf, ref, 10) == 0);
  }

  {
    char buf[] = "1234567890";
    char ref[] = "1231234567";
    assert(cuda::std::memmove(buf + 3, buf, 7) == buf + 3);
    assert(cuda::std::memcmp(buf, ref, 10) == 0);
  }

  {
    char buf[] = "1234567890";
    char ref[] = "5678967890";
    assert(cuda::std::memmove(buf, buf + 4, 5) == buf);
    assert(cuda::std::memcmp(buf, ref, 10) == 0);
  }

  {
    char buf[] = "1234567890";
    char ref[] = "1234897890";
    assert(cuda::std::memmove(buf + 4, buf + 7, 2) == buf + 4);
    assert(cuda::std::memcmp(buf, ref, 10) == 0);
  }

  return true;
}

__host__ __device__ bool test()
{
  test_out_of_place();
  test_in_place();
  return true;
}

int main(int, char**)
{
  test();
  return 0;
}
