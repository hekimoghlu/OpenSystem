/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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

template <size_t ExpectedSize, class OffsetType, class ExtentType, class StrideType>
__host__ __device__ void test(cuda::std::strided_slice<OffsetType, ExtentType, StrideType> slice, size_t expected_size)
{
  using strided_slice = cuda::std::strided_slice<OffsetType, ExtentType, StrideType>;
  assert(slice.offset == 42);
  assert(slice.extent == 1337);
  assert(slice.stride == 7);
  assert(sizeof(strided_slice) == expected_size);
  static_assert(sizeof(strided_slice) == ExpectedSize, "Size mismatch");
}

template <size_t ExpectedSize, class OffsetType, class ExtentType, class StrideType>
__global__ void test_kernel(cuda::std::strided_slice<OffsetType, ExtentType, StrideType> slice, size_t expected_size)
{
  test<ExpectedSize>(slice, expected_size);
}

void test()
{
  { // all non_empty
    using strided_slice = cuda::std::strided_slice<int, int, int>;
    strided_slice slice{42, 1337, 7};
    test<sizeof(strided_slice)>(slice, sizeof(strided_slice));
    test_kernel<sizeof(strided_slice)><<<1, 1>>>(slice, sizeof(strided_slice));
  }

  { // OffsetType empty
    using strided_slice = cuda::std::strided_slice<cuda::std::integral_constant<int, 42>, int, int>;
    strided_slice slice{{}, 1337, 7};
    test<sizeof(strided_slice)>(slice, sizeof(strided_slice));
    test_kernel<sizeof(strided_slice)><<<1, 1>>>(slice, sizeof(strided_slice));
  }

  { // ExtentType empty
    using strided_slice = cuda::std::strided_slice<int, cuda::std::integral_constant<int, 1337>, int>;
    strided_slice slice{42, {}, 7};
    test<sizeof(strided_slice)>(slice, sizeof(strided_slice));
    test_kernel<sizeof(strided_slice)><<<1, 1>>>(slice, sizeof(strided_slice));
  }

  { // StrideType empty
    using strided_slice = cuda::std::strided_slice<int, int, cuda::std::integral_constant<int, 7>>;
    strided_slice slice{42, 1337, {}};
    test<sizeof(strided_slice)>(slice, sizeof(strided_slice));
    test_kernel<sizeof(strided_slice)><<<1, 1>>>(slice, sizeof(strided_slice));
  }

  { // OffsetType + StrideType empty
    using strided_slice =
      cuda::std::strided_slice<cuda::std::integral_constant<int, 42>, int, cuda::std::integral_constant<int, 7>>;
    strided_slice slice{{}, 1337, {}};
    test<sizeof(strided_slice)>(slice, sizeof(strided_slice));
    test_kernel<sizeof(strided_slice)><<<1, 1>>>(slice, sizeof(strided_slice));
  }

  { // All empty
    using strided_slice =
      cuda::std::strided_slice<cuda::std::integral_constant<int, 42>,
                               cuda::std::integral_constant<int, 1337>,
                               cuda::std::integral_constant<int, 7>>;
    strided_slice slice{};
    test<sizeof(strided_slice)>(slice, sizeof(strided_slice));
    // cannot call a kernel with an Empty parameter type
    // test_kernel<sizeof(strided_slice)><<<1, 1>>>(slice, sizeof(strided_slice));
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
