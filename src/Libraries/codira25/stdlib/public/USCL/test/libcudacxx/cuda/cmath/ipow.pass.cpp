/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 3, 2022.
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

#include <uscl/cmath>
#include <uscl/std/cassert>
#include <uscl/std/limits>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

template <class B, class E, class In1, class In2, class Ref>
__host__ __device__ constexpr void test_ipow(In1 base, In2 exp, Ref ref)
{
  if (cuda::std::in_range<B>(base) && cuda::std::in_range<E>(exp) && cuda::std::in_range<B>(ref))
  {
    assert(cuda::ipow(static_cast<B>(base), static_cast<E>(exp)) == static_cast<B>(ref));
  }
}

template <class B, class E>
__host__ __device__ constexpr void test_type()
{
  static_assert(cuda::std::is_same_v<decltype(cuda::ipow(B{}, E{})), B>);
  static_assert(noexcept(cuda::ipow(B{}, E{})));

  test_ipow<B, E>(0, 0, 1);
  test_ipow<B, E>(1, 0, 1);
  test_ipow<B, E>(127, 0, 1);
  test_ipow<B, E>(-127, 0, 1);

  test_ipow<B, E>(1, 1, 1);
  test_ipow<B, E>(1, 2, 1);
  test_ipow<B, E>(1, -1, 1);
  test_ipow<B, E>(-1, 1, -1);
  test_ipow<B, E>(-1, 2, 1);
  test_ipow<B, E>(-1, 3, -1);

  test_ipow<B, E>(2, 2, 4);
  test_ipow<B, E>(2, 12, 4'096);
  test_ipow<B, E>(2, -2, 0);
  test_ipow<B, E>(2, -12, 0);

  test_ipow<B, E>(4, 5, 1'024);
  test_ipow<B, E>(4, 7, 16'384);
  test_ipow<B, E>(4, -5, 0);
  test_ipow<B, E>(128, 5, 34'359'738'368);
  test_ipow<B, E>(1024, 3, 1'073'741'824);

  test_ipow<B, E>(17, 7, 410'338'673);
  test_ipow<B, E>(57, 3, 185'193);
  test_ipow<B, E>(57, -127, 0);
  test_ipow<B, E>(101, 3, 1'030'301);
  test_ipow<B, E>(7891, 2, 62'267'881);
}

template <class B>
__host__ __device__ constexpr void test_type()
{
  test_type<B, signed char>();
  test_type<B, signed short>();
  test_type<B, signed int>();
  test_type<B, signed long>();
  test_type<B, signed long long>();
#if _CCCL_HAS_INT128()
  test_type<B, __int128_t>();
#endif // _CCCL_HAS_INT128()

  test_type<B, unsigned char>();
  test_type<B, unsigned short>();
  test_type<B, unsigned int>();
  test_type<B, unsigned long>();
  test_type<B, unsigned long long>();
#if _CCCL_HAS_INT128()
  test_type<B, __uint128_t>();
#endif // _CCCL_HAS_INT128()
}

__host__ __device__ constexpr bool test()
{
  test_type<signed char>();
  test_type<signed short>();
  test_type<signed int>();
  test_type<signed long>();
  test_type<signed long long>();
#if _CCCL_HAS_INT128()
  test_type<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test_type<unsigned char>();
  test_type<unsigned short>();
  test_type<unsigned int>();
  test_type<unsigned long>();
  test_type<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_type<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
