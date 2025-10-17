/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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
//

// <numeric>

// template<class _M, class _N>
// constexpr common_type_t<_M,_N> gcd(_M __m, _N __n)

#include <uscl/std/cassert>
#include <uscl/std/climits>
#include <uscl/std/cstdint>
#include <uscl/std/numeric>
#include <uscl/std/type_traits>

#include "test_macros.h"

struct TestCase
{
  int x;
  int y;
  int expect;
};

template <typename Input1, typename Input2, typename Output>
__host__ __device__ constexpr bool test0(int in1, int in2, int out)
{
  auto value1 = static_cast<Input1>(in1);
  auto value2 = static_cast<Input2>(in2);
  static_assert(cuda::std::is_same<Output, decltype(cuda::std::gcd(value1, value2))>::value, "");
  static_assert(cuda::std::is_same<Output, decltype(cuda::std::gcd(value2, value1))>::value, "");
  assert(static_cast<Output>(out) == cuda::std::gcd(value1, value2));
  return true;
}

template <typename Input1, typename Input2 = Input1>
__host__ __device__ constexpr void test()
{
  using S1                   = cuda::std::make_signed_t<Input1>;
  using S2                   = cuda::std::make_signed_t<Input2>;
  using U1                   = cuda::std::make_signed_t<Input1>;
  using U2                   = cuda::std::make_signed_t<Input2>;
  bool accumulate            = true;
  constexpr TestCase Cases[] = {
    {0, 0, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}, {2, 3, 1}, {2, 4, 2}, {36, 17, 1}, {36, 18, 18}};
  for (auto TC : Cases)
  {
    { // Test with two signed types
      using Output = cuda::std::common_type_t<S1, S2>;
      accumulate &= test0<S1, S2, Output>(TC.x, TC.y, TC.expect);
      accumulate &= test0<S1, S2, Output>(-TC.x, TC.y, TC.expect);
      accumulate &= test0<S1, S2, Output>(TC.x, -TC.y, TC.expect);
      accumulate &= test0<S1, S2, Output>(-TC.x, -TC.y, TC.expect);
      accumulate &= test0<S2, S1, Output>(TC.x, TC.y, TC.expect);
      accumulate &= test0<S2, S1, Output>(-TC.x, TC.y, TC.expect);
      accumulate &= test0<S2, S1, Output>(TC.x, -TC.y, TC.expect);
      accumulate &= test0<S2, S1, Output>(-TC.x, -TC.y, TC.expect);
    }
    { // test with two unsigned types
      using Output = cuda::std::common_type_t<U1, U2>;
      accumulate &= test0<U1, U2, Output>(TC.x, TC.y, TC.expect);
      accumulate &= test0<U2, U1, Output>(TC.x, TC.y, TC.expect);
    }
    { // Test with mixed signs
      using Output = cuda::std::common_type_t<S1, U2>;
      accumulate &= test0<S1, U2, Output>(TC.x, TC.y, TC.expect);
      accumulate &= test0<U2, S1, Output>(TC.x, TC.y, TC.expect);
      accumulate &= test0<S1, U2, Output>(-TC.x, TC.y, TC.expect);
      accumulate &= test0<U2, S1, Output>(TC.x, -TC.y, TC.expect);
    }
    { // Test with mixed signs
      using Output = cuda::std::common_type_t<S2, U1>;
      accumulate &= test0<S2, U1, Output>(TC.x, TC.y, TC.expect);
      accumulate &= test0<U1, S2, Output>(TC.x, TC.y, TC.expect);
      accumulate &= test0<S2, U1, Output>(-TC.x, TC.y, TC.expect);
      accumulate &= test0<U1, S2, Output>(TC.x, -TC.y, TC.expect);
    }
  }
  assert(accumulate);
}

__host__ __device__ constexpr bool test()
{
  test<signed char>();
  test<short>();
  test<int>();
  test<long>();
  test<long long>();

  test<cuda::std::int8_t>();
  test<cuda::std::int16_t>();
  test<cuda::std::int32_t>();
  test<cuda::std::int64_t>();

  test<signed char, int>();
  test<int, signed char>();
  test<short, int>();
  test<int, short>();
  test<int, long>();
  test<long, int>();
  test<int, long long>();
  test<long long, int>();

  return true;
}

int main(int argc, char**)
{
  test();
  static_assert(test(), "");

  //  LWG#2837
  {
    auto res = cuda::std::gcd(static_cast<cuda::std::int64_t>(1234), INT32_MIN);
    static_assert(cuda::std::is_same<decltype(res), cuda::std::int64_t>::value, "");
    assert(res == 2);
  }

  return 0;
}
