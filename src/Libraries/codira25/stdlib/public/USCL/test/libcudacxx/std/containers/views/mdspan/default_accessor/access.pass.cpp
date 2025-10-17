/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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

// <mdspan>

// constexpr reference access(data_handle_type p, size_t i) const noexcept;
//
// Effects: Equivalent to: return p[i];

#include <uscl/std/cassert>
#include <uscl/std/mdspan>
#include <uscl/std/type_traits>

#include "../MinimalElementType.h"
#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void test_access()
{
  ElementPool<cuda::std::remove_const_t<T>, 10> data;
  T* ptr = data.get_ptr();
  cuda::std::default_accessor<T> acc{};
  for (int i = 0; i < 10; i++)
  {
    static_assert(
      cuda::std::is_same<decltype(acc.access(ptr, i)), typename cuda::std::default_accessor<T>::reference>::value, "");
    static_assert(noexcept(acc.access(ptr, i)));
    assert(&acc.access(ptr, i) == ptr + i);
  }
}

__host__ __device__ constexpr bool test()
{
  test_access<int>();
  test_access<const int>();
  test_access<MinimalElementType>();
  test_access<const MinimalElementType>();
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
