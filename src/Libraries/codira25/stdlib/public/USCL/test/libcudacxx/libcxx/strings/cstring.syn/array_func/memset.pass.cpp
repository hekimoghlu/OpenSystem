/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 24, 2023.
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

#include <uscl/std/__string/constexpr_c_functions.h>
#include <uscl/std/cassert>

template <class T>
__host__ __device__ constexpr void test_memset(T* ptr, T c, cuda::std::size_t n, const T* ref)
{
  assert(cuda::std::__cccl_memset(ptr, c, n) == ptr);
  assert(cuda::std::__cccl_memcmp(ptr, ref, n) == 0);
}

template <class T>
__host__ __device__ constexpr void test_type()
{
  {
    test_memset<T>(nullptr, 1, 0, nullptr);
  }
  {
    constexpr T value = 127;
    T obj{127};
    T ref{value};
    test_memset(&obj, value, 1, &ref);
  }
  {
    constexpr T value = 29;
    T arr[]{127, 46, 7};
    const T ref[]{127, 46, 7};
    test_memset(arr, value, 0, ref);
  }
  {
    constexpr T value = 29;
    T arr[]{127, 46, 7};
    const T ref[]{value, 46, 7};
    test_memset(arr, value, 1, ref);
  }
  {
    constexpr T value = 29;
    T arr[]{127, 46, 7};
    const T ref[]{value, value, 7};
    test_memset(arr, value, 2, ref);
  }
  {
    constexpr T value = 29;
    T arr[]{127, 46, 7};
    const T ref[]{value, value, value};
    test_memset(arr, value, 3, ref);
  }
}

__host__ __device__ constexpr bool test()
{
  test_type<char>();
#if _CCCL_HAS_CHAR8_T()
  test_type<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test_type<char16_t>();
  test_type<char32_t>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
