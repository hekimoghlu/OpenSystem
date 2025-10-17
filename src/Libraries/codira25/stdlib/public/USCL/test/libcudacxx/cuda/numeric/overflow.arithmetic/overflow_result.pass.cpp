/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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

#include <uscl/numeric>
#include <uscl/std/cassert>
#include <uscl/std/cstddef>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void test_member(const T&)
{}

template <class T>
__host__ __device__ constexpr void test_type()
{
  // 1. Test properties of overflow_result
  {
    static_assert(cuda::std::is_standard_layout_v<cuda::overflow_result<T>>);
    static_assert(cuda::std::is_trivially_default_constructible_v<cuda::overflow_result<T>>);
    static_assert(cuda::std::is_trivially_copyable_v<cuda::overflow_result<T>>);
    static_assert(cuda::std::is_trivially_copy_assignable_v<cuda::overflow_result<T>>);
    static_assert(cuda::std::is_trivially_move_constructible_v<cuda::overflow_result<T>>);
    static_assert(cuda::std::is_trivially_move_assignable_v<cuda::overflow_result<T>>);
    static_assert(cuda::std::is_trivially_destructible_v<cuda::overflow_result<T>>);
  }

  // 2. Test members and member types
  {
    cuda::overflow_result<T> r{};
    test_member(r.value);
    test_member(r.overflow);

    static_assert(cuda::std::is_same_v<decltype(r.value), T>);
    static_assert(cuda::std::is_same_v<decltype(r.overflow), bool>);

#if TEST_COMPILER(NVRTC)
    assert(offsetof(cuda::overflow_result<T>, value) == 0);
    assert(offsetof(cuda::overflow_result<T>, overflow) > offsetof(cuda::overflow_result<T>, value));
#else // ^^^ TEST_COMPILER(NVRTC) ^^^ / vvv !TEST_COMPILER(NVRTC) vvv
    static_assert(offsetof(cuda::overflow_result<T>, value) == 0);
    static_assert(offsetof(cuda::overflow_result<T>, overflow) > offsetof(cuda::overflow_result<T>, value));
#endif // !TEST_COMPILER(NVRTC)
  }

  // 3. Test explicit bool conversion
  {
    static_assert(!cuda::std::is_convertible_v<cuda::overflow_result<T>, bool>);

    cuda::overflow_result<T> r{};
    r.overflow = false;
    assert(static_cast<bool>(r) == r.overflow);
    r.overflow = true;
    assert(static_cast<bool>(r) == r.overflow);
  }

  // 4. Test structured bindings
#if __cpp_structured_bindings >= 201606L
  {
    const auto [value, overflow] = cuda::overflow_result<T>{T(1), true};
    assert(value == T(1));
    assert(overflow == true);
  }
#endif // __cpp_structured_bindings >= 201606L
}

__host__ __device__ constexpr bool test()
{
  test_type<signed char>();
  test_type<unsigned char>();
  test_type<short>();
  test_type<unsigned short>();
  test_type<int>();
  test_type<unsigned int>();
  test_type<long>();
  test_type<unsigned long>();
  test_type<long long>();
  test_type<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_type<__int128_t>();
  test_type<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int arg, char** argv)
{
  test();
  static_assert(test());
  return 0;
}
