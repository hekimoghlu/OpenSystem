/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 15, 2023.
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
//
//===----------------------------------------------------------------------===//

// type_traits

// remove_all_extents

#include <uscl/std/type_traits>

#include "test_macros.h"

enum Enum
{
  zero,
  one_
};

template <class T, class U>
__host__ __device__ void test_remove_all_extents()
{
  static_assert(cuda::std::is_same_v<U, typename cuda::std::remove_all_extents<T>::type>);
  static_assert(cuda::std::is_same_v<U, cuda::std::remove_all_extents_t<T>>);
}

int main(int, char**)
{
  test_remove_all_extents<int, int>();
  test_remove_all_extents<const Enum, const Enum>();
  test_remove_all_extents<int[], int>();
  test_remove_all_extents<const int[], const int>();
  test_remove_all_extents<int[3], int>();
  test_remove_all_extents<const int[3], const int>();
  test_remove_all_extents<int[][3], int>();
  test_remove_all_extents<const int[][3], const int>();
  test_remove_all_extents<int[2][3], int>();
  test_remove_all_extents<const int[2][3], const int>();
  test_remove_all_extents<int[1][2][3], int>();
  test_remove_all_extents<const int[1][2][3], const int>();

  return 0;
}
