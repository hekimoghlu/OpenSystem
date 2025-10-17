/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 24, 2025.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <uscl/std/cassert>
#include <uscl/std/mdspan>

#include "test_macros.h"

template <size_t First, size_t Second, size_t Third>
__host__ __device__ void test(cuda::std::extents<size_t, First, Second, Third> ext, size_t expected_size)
{
  using extents = cuda::std::extents<size_t, First, Second, Third>;
  assert(ext.extent(0) == 42);
  assert(ext.extent(1) == 1337);
  assert(ext.extent(2) == 7);
  assert(sizeof(extents) == expected_size);
}

template <size_t First, size_t Second, size_t Third>
__global__ void test_kernel(cuda::std::extents<size_t, First, Second, Third> ext, size_t expected_size)
{
  test(ext, expected_size);
}

void test()
{
  { // all dynamic
    using extents =
      cuda::std::extents<size_t, cuda::std::dynamic_extent, cuda::std::dynamic_extent, cuda::std::dynamic_extent>;
    extents ext{42, 1337, 7};
    test(ext, sizeof(extents));
    test_kernel<<<1, 1>>>(ext, sizeof(extents));
  }

  { // middle static
    using extents = cuda::std::extents<size_t, cuda::std::dynamic_extent, 1337, cuda::std::dynamic_extent>;
    extents ext{42, 7};
    test(ext, sizeof(extents));
    test_kernel<<<1, 1>>>(ext, sizeof(extents));
  }

  { // middle dynamic
    using extents = cuda::std::extents<size_t, 42, cuda::std::dynamic_extent, 7>;
    extents ext{1337};
    test(ext, sizeof(extents));
    test_kernel<<<1, 1>>>(ext, sizeof(extents));
  }

  { // all dynamic
    using extents = cuda::std::extents<size_t, 42, 1337, 7>;
    extents ext{};
    test(ext, sizeof(extents));
    test_kernel<<<1, 1>>>(ext, sizeof(extents));
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
