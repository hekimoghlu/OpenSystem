/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 14, 2023.
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

// is_final

#include <uscl/std/type_traits>

#include "test_macros.h"

struct P final
{};
union U1
{};
union U2 final
{};

template <class T>
__host__ __device__ void test_is_final()
{
  static_assert(cuda::std::is_final<T>::value, "");
  static_assert(cuda::std::is_final<const T>::value, "");
  static_assert(cuda::std::is_final<volatile T>::value, "");
  static_assert(cuda::std::is_final<const volatile T>::value, "");
  static_assert(cuda::std::is_final_v<T>, "");
  static_assert(cuda::std::is_final_v<const T>, "");
  static_assert(cuda::std::is_final_v<volatile T>, "");
  static_assert(cuda::std::is_final_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_is_not_final()
{
  static_assert(!cuda::std::is_final<T>::value, "");
  static_assert(!cuda::std::is_final<const T>::value, "");
  static_assert(!cuda::std::is_final<volatile T>::value, "");
  static_assert(!cuda::std::is_final<const volatile T>::value, "");
  static_assert(!cuda::std::is_final_v<T>, "");
  static_assert(!cuda::std::is_final_v<const T>, "");
  static_assert(!cuda::std::is_final_v<volatile T>, "");
  static_assert(!cuda::std::is_final_v<const volatile T>, "");
}

int main(int, char**)
{
  test_is_not_final<int>();
  test_is_not_final<int*>();
  test_is_final<P>();
  test_is_not_final<P*>();
  test_is_not_final<U1>();
  test_is_not_final<U1*>();
  test_is_final<U2>();
  test_is_not_final<U2*>();

  return 0;
}
