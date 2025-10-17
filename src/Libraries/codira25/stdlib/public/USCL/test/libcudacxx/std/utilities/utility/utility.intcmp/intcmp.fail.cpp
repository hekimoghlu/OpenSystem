/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 2, 2022.
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

// <utility>

// template<class T, class U>
//   constexpr bool cmp_equal(T t, U u) noexcept;

// template<class T, class U>
//   constexpr bool cmp_not_equal(T t, U u) noexcept;

// template<class T, class U>
//   constexpr bool cmp_less(T t, U u) noexcept;

// template<class T, class U>
//   constexpr bool cmp_less_equal(T t, U u) noexcept;

// template<class T, class U>
//   constexpr bool cmp_greater(T t, U u) noexcept;

// template<class T, class U>
//   constexpr bool cmp_greater_equal(T t, U u) noexcept;

// template<class R, class T>
//   constexpr bool in_range(T t) noexcept;

#include <uscl/std/cstddef>
#include <uscl/std/utility>

#include "test_macros.h"

struct NonEmptyT
{
  int val;
  __host__ __device__ NonEmptyT()
      : val(0)
  {}
  __host__ __device__ NonEmptyT(int val)
      : val(val)
  {}
  __host__ __device__ operator int&()
  {
    return val;
  }
  __host__ __device__ operator int() const
  {
    return val;
  }
};

enum ColorT
{
  red,
  green,
  blue
};

struct EmptyT
{};

template <class T>
__host__ __device__ constexpr void test()
{
  cuda::std::cmp_equal(T(), T()); // expected-error 10-11 {{no matching function for call to 'cmp_equal'}}
  cuda::std::cmp_equal(T(), int()); // expected-error 10-11 {{no matching function for call to 'cmp_equal'}}
  cuda::std::cmp_equal(int(), T()); // expected-error 10-11 {{no matching function for call to 'cmp_equal'}}
  cuda::std::cmp_not_equal(T(), T()); // expected-error 10-11 {{no matching function for call to 'cmp_not_equal'}}
  cuda::std::cmp_not_equal(T(), int()); // expected-error 10-11 {{no matching function for call to 'cmp_not_equal'}}
  cuda::std::cmp_not_equal(int(), T()); // expected-error 10-11 {{no matching function for call to 'cmp_not_equal'}}
  cuda::std::cmp_less(T(), T()); // expected-error 10-11 {{no matching function for call to 'cmp_less'}}
  cuda::std::cmp_less(T(), int()); // expected-error 10-11 {{no matching function for call to 'cmp_less'}}
  cuda::std::cmp_less(int(), T()); // expected-error 10-11 {{no matching function for call to 'cmp_less'}}
  cuda::std::cmp_less_equal(T(), T()); // expected-error 10-11 {{no matching function for call to 'cmp_less_equal'}}
  cuda::std::cmp_less_equal(T(), int()); // expected-error 10-11 {{no matching function for call to 'cmp_less_equal'}}
  cuda::std::cmp_less_equal(int(), T()); // expected-error 10-11 {{no matching function for call to 'cmp_less_equal'}}
  cuda::std::cmp_greater(T(), T()); // expected-error 10-11 {{no matching function for call to 'cmp_greater'}}
  cuda::std::cmp_greater(T(), int()); // expected-error 10-11 {{no matching function for call to 'cmp_greater'}}
  cuda::std::cmp_greater(int(), T()); // expected-error 10-11 {{no matching function for call to 'cmp_greater'}}
  cuda::std::cmp_greater_equal(T(), T()); // expected-error 10-11 {{no matching function for call to
                                          // 'cmp_greater_equal'}}
  cuda::std::cmp_greater_equal(T(), int()); // expected-error 10-11 {{no matching function for call to
                                            // 'cmp_greater_equal'}}
  cuda::std::cmp_greater_equal(int(), T()); // expected-error 10-11 {{no matching function for call to
                                            // 'cmp_greater_equal'}}
  cuda::std::in_range<T>(int()); // expected-error 10-11 {{no matching function for call to 'in_range'}}
  cuda::std::in_range<int>(T()); // expected-error 10-11 {{no matching function for call to 'in_range'}}
}
#if _CCCL_HAS_CHAR8_T()
template <class T>
__host__ __device__ constexpr void test_char8t()
{
  cuda::std::cmp_equal(T(), T()); // expected-error 1 {{no matching function for call to 'cmp_equal'}}
  cuda::std::cmp_equal(T(), int()); // expected-error 1 {{no matching function for call to 'cmp_equal'}}
  cuda::std::cmp_equal(int(), T()); // expected-error 1 {{no matching function for call to 'cmp_equal'}}
  cuda::std::cmp_not_equal(T(), T()); // expected-error 1 {{no matching function for call to 'cmp_not_equal'}}
  cuda::std::cmp_not_equal(T(), int()); // expected-error 1 {{no matching function for call to 'cmp_not_equal'}}
  cuda::std::cmp_not_equal(int(), T()); // expected-error 1 {{no matching function for call to 'cmp_not_equal'}}
  cuda::std::cmp_less(T(), T()); // expected-error 1 {{no matching function for call to 'cmp_less'}}
  cuda::std::cmp_less(T(), int()); // expected-error 1 {{no matching function for call to 'cmp_less'}}
  cuda::std::cmp_less(int(), T()); // expected-error 1 {{no matching function for call to 'cmp_less'}}
  cuda::std::cmp_less_equal(T(), T()); // expected-error 1 {{no matching function for call to 'cmp_less_equal'}}
  cuda::std::cmp_less_equal(T(), int()); // expected-error 1 {{no matching function for call to 'cmp_less_equal'}}
  cuda::std::cmp_less_equal(int(), T()); // expected-error 1 {{no matching function for call to 'cmp_less_equal'}}
  cuda::std::cmp_greater(T(), T()); // expected-error 1 {{no matching function for call to 'cmp_greater'}}
  cuda::std::cmp_greater(T(), int()); // expected-error 1 {{no matching function for call to 'cmp_greater'}}
  cuda::std::cmp_greater(int(), T()); // expected-error 1 {{no matching function for call to 'cmp_greater'}}
  cuda::std::cmp_greater_equal(T(), T()); // expected-error 1 {{no matching function for call to 'cmp_greater_equal'}}
  cuda::std::cmp_greater_equal(T(), int()); // expected-error 1 {{no matching function for call to 'cmp_greater_equal'}}
  cuda::std::cmp_greater_equal(int(), T()); // expected-error 1 {{no matching function for call to 'cmp_greater_equal'}}
  cuda::std::in_range<T>(int()); // expected-error 1 {{no matching function for call to 'in_range'}}
  cuda::std::in_range<int>(T()); // expected-error 1 {{no matching function for call to 'in_range'}}
}
#endif // _CCCL_HAS_CHAR8_T()

template <class T>
__host__ __device__ constexpr void test_uchars()
{
  cuda::std::cmp_equal(T(), T()); // expected-error 2 {{no matching function for call to 'cmp_equal'}}
  cuda::std::cmp_equal(T(), int()); // expected-error 2 {{no matching function for call to 'cmp_equal'}}
  cuda::std::cmp_equal(int(), T()); // expected-error 2 {{no matching function for call to 'cmp_equal'}}
  cuda::std::cmp_not_equal(T(), T()); // expected-error 2 {{no matching function for call to 'cmp_not_equal'}}
  cuda::std::cmp_not_equal(T(), int()); // expected-error 2 {{no matching function for call to 'cmp_not_equal'}}
  cuda::std::cmp_not_equal(int(), T()); // expected-error 2 {{no matching function for call to 'cmp_not_equal'}}
  cuda::std::cmp_less(T(), T()); // expected-error 2 {{no matching function for call to 'cmp_less'}}
  cuda::std::cmp_less(T(), int()); // expected-error 2 {{no matching function for call to 'cmp_less'}}
  cuda::std::cmp_less(int(), T()); // expected-error 2 {{no matching function for call to 'cmp_less'}}
  cuda::std::cmp_less_equal(T(), T()); // expected-error 2 {{no matching function for call to 'cmp_less_equal'}}
  cuda::std::cmp_less_equal(T(), int()); // expected-error 2 {{no matching function for call to 'cmp_less_equal'}}
  cuda::std::cmp_less_equal(int(), T()); // expected-error 2 {{no matching function for call to 'cmp_less_equal'}}
  cuda::std::cmp_greater(T(), T()); // expected-error 2 {{no matching function for call to 'cmp_greater'}}
  cuda::std::cmp_greater(T(), int()); // expected-error 2 {{no matching function for call to 'cmp_greater'}}
  cuda::std::cmp_greater(int(), T()); // expected-error 2 {{no matching function for call to 'cmp_greater'}}
  cuda::std::cmp_greater_equal(T(), T()); // expected-error 2 {{no matching function for call to 'cmp_greater_equal'}}
  cuda::std::cmp_greater_equal(T(), int()); // expected-error 2 {{no matching function for call to 'cmp_greater_equal'}}
  cuda::std::cmp_greater_equal(int(), T()); // expected-error 2 {{no matching function for call to 'cmp_greater_equal'}}
  cuda::std::in_range<T>(int()); // expected-error 2 {{no matching function for call to 'in_range'}}
  cuda::std::in_range<int>(T()); // expected-error 2 {{no matching function for call to 'in_range'}}
}

int main(int, char**)
{
  test<bool>();
  test<char>();
  test<wchar_t>();
  test<float>();
  test<double>();
  test<long double>();
  test<cuda::std::byte>();
  test<NonEmptyT>();
  test<ColorT>();
  test<cuda::std::nullptr_t>();
  test<EmptyT>();

#if _CCCL_HAS_CHAR8_T()
  test_char8t<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()

  test_uchars<char16_t>();
  test_uchars<char32_t>();

  return 0;
}
