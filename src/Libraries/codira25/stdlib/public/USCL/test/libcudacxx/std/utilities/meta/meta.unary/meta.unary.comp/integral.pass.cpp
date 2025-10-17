/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 20, 2025.
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

// integral

#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_integral_imp()
{
  static_assert(!cuda::std::is_reference<T>::value, "");
  static_assert(cuda::std::is_arithmetic<T>::value, "");
  static_assert(cuda::std::is_fundamental<T>::value, "");
  static_assert(cuda::std::is_object<T>::value, "");
  static_assert(cuda::std::is_scalar<T>::value, "");
  static_assert(!cuda::std::is_compound<T>::value, "");
  static_assert(!cuda::std::is_member_pointer<T>::value, "");
}

template <class T>
__host__ __device__ void test_integral()
{
  test_integral_imp<T>();
  test_integral_imp<const T>();
  test_integral_imp<volatile T>();
  test_integral_imp<const volatile T>();
}

int main(int, char**)
{
  test_integral<bool>();
  test_integral<char>();
  test_integral<signed char>();
  test_integral<unsigned char>();
  test_integral<wchar_t>();
  test_integral<short>();
  test_integral<unsigned short>();
  test_integral<int>();
  test_integral<unsigned int>();
  test_integral<long>();
  test_integral<unsigned long>();
  test_integral<long long>();
  test_integral<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_integral<__int128_t>();
  test_integral<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return 0;
}
