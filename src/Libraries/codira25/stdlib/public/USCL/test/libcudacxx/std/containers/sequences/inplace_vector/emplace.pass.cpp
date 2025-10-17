/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 6, 2025.
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
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <uscl/std/__algorithm_>
#include <uscl/std/array>
#include <uscl/std/cassert>
#include <uscl/std/initializer_list>
#include <uscl/std/inplace_vector>
#include <uscl/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

#if TEST_HAS_EXCEPTIONS()
#  include <stdexcept>
#endif // TEST_HAS_EXCEPTIONS()

template <class T>
__host__ __device__ constexpr void test()
{
  constexpr size_t max_capacity         = 6ull;
  using inplace_vector                  = cuda::std::inplace_vector<T, max_capacity>;
  const cuda::std::array<T, 6> expected = {T(0), T(1), T(2), T(3), T(4), T(5)};

  { // inplace_vector<T, N>::emplace(iter, args...)
    inplace_vector vec = {T(0), T(1), T(2), T(4), T(5)};
    const auto res     = vec.emplace(vec.begin() + 3, 3);
    static_assert(cuda::std::is_same<decltype(res), const typename inplace_vector::iterator>::value, "");
    assert(equal_range(vec, expected));
    assert(res == vec.begin() + 3);
  }

  { // inplace_vector<T, N>::emplace(const_iter, args...)
    inplace_vector vec = {T(0), T(1), T(2), T(4), T(5)};
    const auto res     = vec.emplace(vec.cbegin() + 3, 3);
    static_assert(cuda::std::is_same<decltype(res), const typename inplace_vector::iterator>::value, "");
    assert(equal_range(vec, expected));
    assert(res == vec.cbegin() + 3);
  }

  { // inplace_vector<T, N>::emplace_back(args...)
    inplace_vector vec = {T(0), T(1), T(2), T(3), T(4)};
    auto&& res         = vec.emplace_back(5);
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::reference>::value, "");
    assert(equal_range(vec, expected));
    assert(res == T(5));
    res = T(6);
    assert(res == T(6));
  }

  { // inplace_vector<T, N>::push_back(const T&)
    const T to_be_pushed = 5;
    inplace_vector vec   = {T(0), T(1), T(2), T(3), T(4)};
    auto&& res           = vec.push_back(to_be_pushed);
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::reference>::value, "");
    assert(equal_range(vec, expected));
    assert(res == T(5));
    res = T(6);
    assert(res == T(6));
  }

  { // inplace_vector<T, N>::push_back(T&&)
    inplace_vector vec = {T(0), T(1), T(2), T(3), T(4)};
    auto&& res         = vec.push_back(T(5));
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::reference>::value, "");
    assert(equal_range(vec, expected));
    assert(res == T(5));
    res = T(6);
    assert(res == T(6));
  }

  { // inplace_vector<T, 0>::try_emplace_back(args...)
    cuda::std::inplace_vector<T, 0> vec{};
    auto res = vec.try_emplace_back(5);
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::pointer>::value, "");
    assert(vec.empty());
    assert(res == nullptr);
  }

  { // inplace_vector<T, N>::try_emplace_back(args...)
    inplace_vector vec = {T(0), T(1), T(2), T(3), T(4)};
    auto res           = vec.try_emplace_back(5);
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::pointer>::value, "");
    assert(equal_range(vec, expected));
    assert(*res == T(5));
    *res = T(6);
    assert(*res == T(6));
  }

  { // inplace_vector<T, N>::try_emplace_back(args...), at capacity
    inplace_vector vec = {T(0), T(1), T(2), T(3), T(4), T(5)};
    auto res           = vec.try_emplace_back(6);
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::pointer>::value, "");
    assert(equal_range(vec, expected));
    assert(res == nullptr);
  }

  { // inplace_vector<T, 0>::try_push_back(const T&)
    const T to_be_pushed = 5;
    cuda::std::inplace_vector<T, 0> vec{};
    auto res = vec.try_push_back(to_be_pushed);
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::pointer>::value, "");
    assert(vec.empty());
    assert(res == nullptr);
  }

  { // inplace_vector<T, N>::try_push_back(const T&)
    const T to_be_pushed = 5;
    inplace_vector vec   = {T(0), T(1), T(2), T(3), T(4)};
    auto res             = vec.try_push_back(to_be_pushed);
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::pointer>::value, "");
    assert(equal_range(vec, expected));
    assert(*res == T(5));
    *res = T(6);
    assert(*res == T(6));
  }

  { // inplace_vector<T, N>::try_push_back(const T&), at capacity
    const T to_be_pushed = 6;
    inplace_vector vec   = {T(0), T(1), T(2), T(3), T(4), T(5)};
    auto res             = vec.try_push_back(to_be_pushed);
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::pointer>::value, "");
    assert(equal_range(vec, expected));
    assert(res == nullptr);
  }

  { // inplace_vector<T, 0>::try_push_back(T&&)
    cuda::std::inplace_vector<T, 0> vec{};
    auto res = vec.try_push_back(5);
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::pointer>::value, "");
    assert(vec.empty());
    assert(res == nullptr);
  }

  { // inplace_vector<T, N>::try_push_back(T&&)
    inplace_vector vec = {T(0), T(1), T(2), T(3), T(4)};
    auto res           = vec.try_push_back(T(5));
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::pointer>::value, "");
    assert(equal_range(vec, expected));
    assert(*res == T(5));
    *res = T(6);
    assert(*res == T(6));
  }

  { // inplace_vector<T, N>::try_push_back(T&&), at capacity
    inplace_vector vec = {T(0), T(1), T(2), T(3), T(4), T(5)};
    auto res           = vec.try_push_back(T(6));
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::pointer>::value, "");
    assert(equal_range(vec, expected));
    assert(res == nullptr);
  }

  { // inplace_vector<T, N>::unchecked_emplace_back(args...)
    inplace_vector vec = {T(0), T(1), T(2), T(3), T(4)};
    auto&& res         = vec.unchecked_emplace_back(5);
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::reference>::value, "");
    assert(equal_range(vec, expected));
    assert(res == T(5));
    res = T(6);
    assert(res == T(6));
  }

  { // inplace_vector<T, N>::unchecked_push_back(const T&)
    const T to_be_pushed = 5;
    inplace_vector vec   = {T(0), T(1), T(2), T(3), T(4)};
    auto&& res           = vec.unchecked_push_back(to_be_pushed);
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::reference>::value, "");
    assert(equal_range(vec, expected));
    assert(res == T(5));
    res = T(6);
    assert(res == T(6));
  }

  { // inplace_vector<T, N>::unchecked_push_back(T&&)
    inplace_vector vec = {T(0), T(1), T(2), T(3), T(4)};
    auto&& res         = vec.unchecked_push_back(T(5));
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::reference>::value, "");
    assert(equal_range(vec, expected));
    assert(res == T(5));
    res = T(6);
    assert(res == T(6));
  }
}

__host__ __device__ constexpr bool test()
{
  test<int>();
  test<Trivial>();

  if (!cuda::std::is_constant_evaluated())
  {
    test<NonTrivial>();
    test<NonTrivialDestructor>();
    test<ThrowingDefaultConstruct>();
  }

  return true;
}

#if TEST_HAS_EXCEPTIONS()
void test_exceptions()
{ // emplace and friends throw std::bad_alloc when out of capacity
  using empty_vec = cuda::std::inplace_vector<int, 0>;
  {
    empty_vec empty{};
    try
    {
      auto emplace = empty.emplace_back(5);
      unused(emplace);
      assert(false);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      assert(false);
    }

    try
    {
      const int input       = 5;
      auto push_back_lvalue = empty.push_back(input);
      unused(push_back_lvalue);
      assert(false);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      assert(false);
    }

    try
    {
      auto push_back_rvalue = empty.push_back(5);
      unused(push_back_rvalue);
      assert(false);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      assert(false);
    }
  }

  using small_vec = cuda::std::inplace_vector<int, 5>;
  {
    small_vec full{0, 1, 2, 3, 4};
    try
    {
      auto emplace = full.emplace_back(5);
      unused(emplace);
      assert(false);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      assert(false);
    }

    try
    {
      const int input       = 5;
      auto push_back_lvalue = full.push_back(input);
      unused(push_back_lvalue);
      assert(false);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      assert(false);
    }

    try
    {
      auto push_back_rvalue = full.push_back(5);
      unused(push_back_rvalue);
      assert(false);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      assert(false);
    }
  }
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED

#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
