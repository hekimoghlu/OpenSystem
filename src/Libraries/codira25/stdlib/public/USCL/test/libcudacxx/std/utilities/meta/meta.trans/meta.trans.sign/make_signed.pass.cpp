/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 1, 2023.
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

// make_signed

#include <uscl/std/type_traits>

#include "test_macros.h"

enum Enum
{
  zero,
  one_
};

enum BigEnum : unsigned long long // MSVC's ABI doesn't follow the Standard
{
  bigzero,
  big = 0xFFFFFFFFFFFFFFFFULL
};

#if _CCCL_HAS_INT128()
enum HugeEnum : __uint128_t
{
  hugezero
};
#endif // _CCCL_HAS_INT128()

template <class T, class U>
__host__ __device__ void test_make_signed()
{
  static_assert(cuda::std::is_same_v<U, typename cuda::std::make_signed<T>::type>);
  static_assert(cuda::std::is_same_v<U, cuda::std::make_signed_t<T>>);
}

int main(int, char**)
{
  test_make_signed<signed char, signed char>();
  test_make_signed<unsigned char, signed char>();
  test_make_signed<char, signed char>();
  test_make_signed<short, signed short>();
  test_make_signed<unsigned short, signed short>();
  test_make_signed<int, signed int>();
  test_make_signed<unsigned int, signed int>();
  test_make_signed<long, signed long>();
  test_make_signed<unsigned long, long>();
  test_make_signed<long long, signed long long>();
  test_make_signed<unsigned long long, signed long long>();
  test_make_signed<wchar_t, cuda::std::conditional<sizeof(wchar_t) == 4, int, short>::type>();
  test_make_signed<const wchar_t, cuda::std::conditional<sizeof(wchar_t) == 4, const int, const short>::type>();
  test_make_signed<const Enum, cuda::std::conditional<sizeof(Enum) == sizeof(int), const int, const signed char>::type>();
  test_make_signed<BigEnum, cuda::std::conditional<sizeof(long) == 4, long long, long>::type>();
#if _CCCL_HAS_INT128()
  test_make_signed<__int128_t, __int128_t>();
  test_make_signed<__uint128_t, __int128_t>();
  test_make_signed<HugeEnum, __int128_t>();
#endif // _CCCL_HAS_INT128()

  return 0;
}
