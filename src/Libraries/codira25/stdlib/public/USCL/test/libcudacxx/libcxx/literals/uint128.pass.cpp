/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 20, 2024.
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

#include <uscl/std/cassert>

#include "literal.h"

// nvcc complains about the u128 literal constants being too large
_CCCL_DIAG_SUPPRESS_NVCC(23)

#if _CCCL_HAS_INT128()

using namespace test_integer_literals;

__host__ __device__ constexpr void test_binary()
{
  assert(0b0_u128 == 0b0);
  assert(0b1_u128 == 0b1);
  assert(0b1111'1111_u128 == 0b1111'1111);
  assert(0b1111'1111'1111'1111_u128 == 0b1111'1111'1111'1111);
  assert(0b0111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111_u128
         == 0b0111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111);

  assert(0B1 == 0b1);
}

__host__ __device__ constexpr void test_octal()
{
  assert(00_u128 == 00);
  assert(01_u128 == 01);
  assert(07_u128 == 07);
  assert(017_u128 == 017);
  assert(0377_u128 == 0377);
  assert(01'777'777'777'777'777'777'777'375_u128 == (__uint128_t{~0ull} << 9) + 0'375);
}

__host__ __device__ constexpr void test_decimal()
{
  assert(0_u128 == 0);
  assert(1_u128 == 1);
  assert(123_u128 == 123);
  assert(123'456'789_u128 == 123'456'789);
  assert(9'223'372'036'854'775'807_u128 == 9'223'372'036'854'775'807);
  assert(12'345'678'901'234'567'890_u128 == 12'345'678'901'234'567'890ull);
}

__host__ __device__ constexpr void test_hexadecimal()
{
  assert(0x0_u128 == 0x0);
  assert(0x1_u128 == 0x1);
  assert(0xF_u128 == 0xF);
  assert(0xFF_u128 == 0xFF);
  assert(0xFFFF_u128 == 0xFFFF);
  assert(0x7FFF'FFFF'FFFF'FFFF_u128 == 0x7FFF'FFFF'FFFF'FFFF);
  assert(0xFF'FFFF'FFFF'FFFF'FFFF_u128 == (__uint128_t{0xFFFF'FFFF'FFFF'FFFF} << 8) + 0xff);
  assert(0x7FFF'FFFF'FFFF'FFFF'FFFF'FFFF'FFFF'FFFF_u128 == ~__uint128_t{0} >> 1);
  assert(0xFFFF'FFFF'FFFF'FFFF'FFFF'FFFF'FFFF'FFFF_u128
         == (__uint128_t{0xFFFF'FFFF'FFFF'FFFF} << 64) + 0xFFFF'FFFF'FFFF'FFFF);

  assert(0Xabcdef_u128 == 0xABCDEF);
}

__host__ __device__ constexpr bool test()
{
  test_binary();
  test_octal();
  test_decimal();
  test_hexadecimal();

  return true;
}

#endif // _CCCL_HAS_INT128()

int main(int, char**)
{
#if _CCCL_HAS_INT128()
  test();
  static_assert(test());
#endif // _CCCL_HAS_INT128()
  return 0;
}
