/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 29, 2024.
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
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// template <class T>
//   constexpr T bit_floor(T x) noexcept;

// Returns: If x == 0, 0; otherwise the maximal value y such that bit_floor(y) is true and y <= x.
// Remarks: This function shall not participate in overload resolution unless
//	T is an unsigned integer type

#include <uscl/std/bit>
#include <uscl/std/cassert>
#include <uscl/std/cstdint>
#include <uscl/std/type_traits>

#include "test_macros.h"

class A
{};
enum E1 : unsigned char
{
  rEd
};
enum class E2 : unsigned char
{
  red
};

template <typename T>
__host__ __device__ constexpr bool constexpr_test()
{
  return cuda::std::bit_floor(T(0)) == T(0) && cuda::std::bit_floor(T(1)) == T(1) && cuda::std::bit_floor(T(2)) == T(2)
      && cuda::std::bit_floor(T(3)) == T(2) && cuda::std::bit_floor(T(4)) == T(4) && cuda::std::bit_floor(T(5)) == T(4)
      && cuda::std::bit_floor(T(6)) == T(4) && cuda::std::bit_floor(T(7)) == T(4) && cuda::std::bit_floor(T(8)) == T(8)
      && cuda::std::bit_floor(T(9)) == T(8);
}

template <typename T>
__host__ __device__ void runtime_test()
{
  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::bit_floor(T(0)))>);
  static_assert(noexcept(cuda::std::bit_floor(T(0))));

  assert(cuda::std::bit_floor(T(121)) == T(64));
  assert(cuda::std::bit_floor(T(122)) == T(64));
  assert(cuda::std::bit_floor(T(123)) == T(64));
  assert(cuda::std::bit_floor(T(124)) == T(64));
  assert(cuda::std::bit_floor(T(125)) == T(64));
  assert(cuda::std::bit_floor(T(126)) == T(64));
  assert(cuda::std::bit_floor(T(127)) == T(64));
  assert(cuda::std::bit_floor(T(128)) == T(128));
  assert(cuda::std::bit_floor(T(129)) == T(128));
  assert(cuda::std::bit_floor(T(130)) == T(128));
}

int main(int, char**)
{
  static_assert(constexpr_test<unsigned char>(), "");
  static_assert(constexpr_test<unsigned short>(), "");
  static_assert(constexpr_test<unsigned>(), "");
  static_assert(constexpr_test<unsigned long>(), "");
  static_assert(constexpr_test<unsigned long long>(), "");

  static_assert(constexpr_test<uint8_t>(), "");
  static_assert(constexpr_test<uint16_t>(), "");
  static_assert(constexpr_test<uint32_t>(), "");
  static_assert(constexpr_test<uint64_t>(), "");
  static_assert(constexpr_test<size_t>(), "");
  static_assert(constexpr_test<uintmax_t>(), "");
  static_assert(constexpr_test<uintptr_t>(), "");

#if _CCCL_HAS_INT128()
  static_assert(constexpr_test<__uint128_t>(), "");
#endif // _CCCL_HAS_INT128()

  runtime_test<unsigned char>();
  runtime_test<unsigned>();
  runtime_test<unsigned short>();
  runtime_test<unsigned long>();
  runtime_test<unsigned long long>();

  runtime_test<uint8_t>();
  runtime_test<uint16_t>();
  runtime_test<uint32_t>();
  runtime_test<uint64_t>();
  runtime_test<size_t>();
  runtime_test<uintmax_t>();
  runtime_test<uintptr_t>();

#if _CCCL_HAS_INT128()
  runtime_test<__uint128_t>();

  {
    __uint128_t val = 128;
    val <<= 32;
    assert(cuda::std::bit_floor(val - 1) == val / 2);
    assert(cuda::std::bit_floor(val) == val);
    assert(cuda::std::bit_floor(val + 1) == val);
    val <<= 2;
    assert(cuda::std::bit_floor(val - 1) == val / 2);
    assert(cuda::std::bit_floor(val) == val);
    assert(cuda::std::bit_floor(val + 1) == val);
    val <<= 3;
    assert(cuda::std::bit_floor(val - 1) == val / 2);
    assert(cuda::std::bit_floor(val) == val);
    assert(cuda::std::bit_floor(val + 1) == val);
  }
#endif // _CCCL_HAS_INT128()

  return 0;
}
