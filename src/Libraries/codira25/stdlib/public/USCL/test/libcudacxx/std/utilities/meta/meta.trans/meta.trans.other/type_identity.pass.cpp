/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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

// type_identity

#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_type_identity()
{
  static_assert(cuda::std::is_same_v<T, typename cuda::std::type_identity<T>::type>);
  static_assert(cuda::std::is_same_v<T, cuda::std::type_identity_t<T>>);
}

int main(int, char**)
{
  test_type_identity<void>();
  test_type_identity<int>();
  test_type_identity<const volatile int>();
  test_type_identity<int*>();
  test_type_identity<int[3]>();
  test_type_identity<const int[3]>();

  test_type_identity<void (*)()>();
  test_type_identity<int(int) const>();
  test_type_identity<int(int) volatile>();
  test_type_identity<int(int) &>();
  test_type_identity<int(int) &&>();

  return 0;
}
