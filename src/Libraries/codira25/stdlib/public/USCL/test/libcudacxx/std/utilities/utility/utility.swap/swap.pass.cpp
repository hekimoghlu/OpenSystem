/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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

// template<class T>
//   requires MoveAssignable<T> && MoveConstructible<T>
//   void
//   swap(T& a, T& b);

#include <uscl/std/__memory_>
#include <uscl/std/cassert>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

#if !TEST_COMPILER(NVRTC)
#  include <memory>
#  include <utility>
#endif // !TEST_COMPILER(NVRTC)

struct CopyOnly
{
  __host__ __device__ CopyOnly() {}
  __host__ __device__ CopyOnly(CopyOnly const&) noexcept {}
  __host__ __device__ CopyOnly& operator=(CopyOnly const&)
  {
    return *this;
  }
};

struct MoveOnly
{
  __host__ __device__ MoveOnly() {}
  __host__ __device__ MoveOnly(MoveOnly&&) {}
  __host__ __device__ MoveOnly& operator=(MoveOnly&&) noexcept
  {
    return *this;
  }
};

struct NoexceptMoveOnly
{
  __host__ __device__ NoexceptMoveOnly() {}
  __host__ __device__ NoexceptMoveOnly(NoexceptMoveOnly&&) noexcept {}
  __host__ __device__ NoexceptMoveOnly& operator=(NoexceptMoveOnly&&) noexcept
  {
    return *this;
  }
};

struct NotMoveConstructible
{
  __host__ __device__ NotMoveConstructible& operator=(NotMoveConstructible&&)
  {
    return *this;
  }

private:
  __host__ __device__ NotMoveConstructible(NotMoveConstructible&&);
};

struct NotMoveAssignable
{
  __host__ __device__ NotMoveAssignable(NotMoveAssignable&&);

private:
  __host__ __device__ NotMoveAssignable& operator=(NotMoveAssignable&&);
};

template <class Tp>
__host__ __device__ auto can_swap_test(int)
  -> decltype(cuda::std::swap(cuda::std::declval<Tp>(), cuda::std::declval<Tp>()));

template <class Tp>
__host__ __device__ auto can_swap_test(...) -> cuda::std::false_type;

template <class Tp>
__host__ __device__ constexpr bool can_swap()
{
  return cuda::std::is_same<decltype(can_swap_test<Tp>(0)), void>::value;
}

__host__ __device__ constexpr bool test_swap_constexpr()
{
  int i = 1;
  int j = 2;
  cuda::std::swap(i, j);
  return i == 2 && j == 1;
}

template <class T>
struct swap_with_friend
{
  __host__ __device__ friend void swap(swap_with_friend&, swap_with_friend&) {}
};

template <typename T>
__host__ __device__ void test_ambiguous_std()
{
  // clang-format off
  NV_IF_TARGET(NV_IS_HOST, (
    // fully qualified calls
    {
      T i = {};
      T j = {};
      cuda::std::swap(i,j);
    }
  ))
#if !TEST_COMPILER(NVRTC)
  NV_IF_TARGET(NV_IS_HOST, (
    {
      T i = {};
      T j = {};
      std::swap(i,j);
    }
  ))
#endif // !TEST_COMPILER(NVRTC)
  NV_IF_TARGET(NV_IS_HOST, (
    // ADL calls
    {
      T i = {};
      T j = {};
      swap(i,j);
    }
  ))
#if !TEST_COMPILER(NVRTC)
  NV_IF_TARGET(NV_IS_HOST, (
    {
      T i = {};
      T j = {};
      using cuda::std::swap;
      swap(i,j);
    }
    {
      T i = {};
      T j = {};
      using std::swap;
      swap(i,j);
    }
    {
      T i = {};
      T j = {};
      using std::swap;
      using cuda::std::swap;
      swap(i,j);
    }
  ))
  // clang-format on
#endif // !TEST_COMPILER(NVRTC)
}

int main(int, char**)
{
  {
    int i = 1;
    int j = 2;
    cuda::std::swap(i, j);
    assert(i == 2);
    assert(j == 1);
  }
  {
    cuda::std::unique_ptr<int> i(new int(1));
    cuda::std::unique_ptr<int> j(new int(2));
    cuda::std::swap(i, j);
    assert(*i == 2);
    assert(*j == 1);
  }
  {
    // test that the swap
    static_assert(can_swap<CopyOnly&>(), "");
    static_assert(can_swap<MoveOnly&>(), "");
    static_assert(can_swap<NoexceptMoveOnly&>(), "");

    static_assert(!can_swap<NotMoveConstructible&>(), "");
    static_assert(!can_swap<NotMoveAssignable&>(), "");

    CopyOnly c;
    MoveOnly m;
    NoexceptMoveOnly nm;
    static_assert(!noexcept(cuda::std::swap(c, c)), "");
    static_assert(!noexcept(cuda::std::swap(m, m)), "");
    static_assert(noexcept(cuda::std::swap(nm, nm)), "");
  }

  static_assert(test_swap_constexpr(), "");

  test_ambiguous_std<cuda::std::pair<int, int>>(); // has cuda::std::swap overload
#if !TEST_COMPILER(NVRTC)
  test_ambiguous_std<::std::pair<int, int>>(); // has std::swap overload
  test_ambiguous_std<cuda::std::pair<::std::pair<int, int>, int>>(); // has std:: and cuda::std as associated namespaces
  test_ambiguous_std<::std::allocator<char>>(); // no std::swap overload

  // Ensure that we do not SFINAE swap out if there is a free function as that will take precedent
  test_ambiguous_std<swap_with_friend<::std::pair<int, int>>>();
#endif // !TEST_COMPILER(NVRTC)

#if !TEST_COMPILER(NVRTC)
  static_assert(cuda::std::is_swappable<cuda::std::pair<::std::pair<int, int>, int>>::value, "");
  static_assert(cuda::std::is_swappable<swap_with_friend<::std::pair<int, int>>>::value, "");
#endif // !TEST_COMPILER(NVRTC)

  return 0;
}
