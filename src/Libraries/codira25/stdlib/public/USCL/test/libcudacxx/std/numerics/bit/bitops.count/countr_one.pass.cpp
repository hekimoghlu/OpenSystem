/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 30, 2024.
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
//   constexpr int countr_one(T x) noexcept;

// Returns: The number of consecutive 1 bits, starting from the least significant bit.
//   [ Note: Returns N if x == cuda::std::numeric_limits<T>::max(). ]
//
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
  static_assert(cuda::std::countr_one(T(2)) == 0, "");
  static_assert(cuda::std::countr_one(T(3)) == 2, "");
  static_assert(cuda::std::countr_one(T(4)) == 0, "");
  static_assert(cuda::std::countr_one(T(5)) == 1, "");
  static_assert(cuda::std::countr_one(T(6)) == 0, "");
  static_assert(cuda::std::countr_one(T(7)) == 3, "");
  static_assert(cuda::std::countr_one(T(8)) == 0, "");
  static_assert(cuda::std::countr_one(T(9)) == 1, "");
  static_assert(cuda::std::countr_one(T(0)) == 0, "");
  static_assert(cuda::std::countr_one(T(1)) == 1, "");
  static_assert(cuda::std::countr_one(cuda::std::numeric_limits<T>::max()) == cuda::std::numeric_limits<T>::digits, "");

  return true;
}

template <typename T>
__host__ __device__ inline void assert_countr_one(T val, int expected)
{
  volatile auto v = val;
  assert(cuda::std::countr_one(v) == expected);
}

template <typename T>
__host__ __device__ void runtime_test()
{
  static_assert(cuda::std::is_same_v<int, decltype(cuda::std::countr_one(T(0)))>);
  static_assert(noexcept(cuda::std::countr_one(T(0))));

  assert_countr_one(T(121), 1);
  assert_countr_one(T(122), 0);
  assert_countr_one(T(123), 2);
  assert_countr_one(T(124), 0);
  assert_countr_one(T(125), 1);
  assert_countr_one(T(126), 0);
  assert_countr_one(T(127), 7);
  assert_countr_one(T(128), 0);
  assert_countr_one(T(129), 1);
  assert_countr_one(T(130), 0);
}

int main(int, char**)
{
  constexpr_test<unsigned char>();
  constexpr_test<unsigned short>();
  constexpr_test<unsigned>();
  constexpr_test<unsigned long>();
  constexpr_test<unsigned long long>();

  constexpr_test<uint8_t>();
  constexpr_test<uint16_t>();
  constexpr_test<uint32_t>();
  constexpr_test<uint64_t>();
  constexpr_test<size_t>();
  constexpr_test<uintmax_t>();
  constexpr_test<uintptr_t>();

#if _CCCL_HAS_INT128()
  constexpr_test<__uint128_t>();
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
    assert(cuda::std::countr_one(val - 1) == 39);
    assert(cuda::std::countr_one(val) == 0);
    assert(cuda::std::countr_one(val + 1) == 1);
    val <<= 2;
    assert(cuda::std::countr_one(val - 1) == 41);
    assert(cuda::std::countr_one(val) == 0);
    assert(cuda::std::countr_one(val + 1) == 1);
    val <<= 3;
    assert(cuda::std::countr_one(val - 1) == 44);
    assert(cuda::std::countr_one(val) == 0);
    assert(cuda::std::countr_one(val + 1) == 1);
  }
#endif // _CCCL_HAS_INT128()

  return 0;
}
