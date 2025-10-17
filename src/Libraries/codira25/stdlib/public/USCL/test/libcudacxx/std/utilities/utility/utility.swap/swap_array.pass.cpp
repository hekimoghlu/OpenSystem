/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 23, 2023.
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

// template<ValueType T, size_t N>
//   requires Swappable<T>
//   void
//   swap(T (&a)[N], T (&b)[N]);

#include <uscl/std/__algorithm_>
#include <uscl/std/__memory_>
#include <uscl/std/array>
#include <uscl/std/cassert>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

#if !TEST_COMPILER(NVRTC)
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
  __host__ __device__ NotMoveConstructible() {}
  __host__ __device__ NotMoveConstructible& operator=(NotMoveConstructible&&)
  {
    return *this;
  }

private:
  __host__ __device__ NotMoveConstructible(NotMoveConstructible&&);
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
  int i[3] = {1, 2, 3};
  int j[3] = {4, 5, 6};
  cuda::std::swap(i, j);
  return i[0] == 4 && i[1] == 5 && i[2] == 6 && j[0] == 1 && j[1] == 2 && j[2] == 3;
}

__host__ __device__ void test_ambiguous_std()
{
#if !TEST_COMPILER(NVRTC)
  // clang-format off
  NV_IF_TARGET(NV_IS_HOST, (
    cuda::std::pair<::std::pair<int, int>, int> i[3] = {};
    cuda::std::pair<::std::pair<int, int>, int> j[3] = {};
    swap(i,j);
  ))
  // clang-format on
#endif // !TEST_COMPILER(NVRTC)
}

int main(int, char**)
{
  {
    int i[3] = {1, 2, 3};
    int j[3] = {4, 5, 6};
    cuda::std::swap(i, j);
    assert(i[0] == 4);
    assert(i[1] == 5);
    assert(i[2] == 6);
    assert(j[0] == 1);
    assert(j[1] == 2);
    assert(j[2] == 3);
  }
  {
    cuda::std::unique_ptr<int> i[3];
    for (int k = 0; k < 3; ++k)
    {
      i[k].reset(new int(k + 1));
    }
    cuda::std::unique_ptr<int> j[3];
    for (int k = 0; k < 3; ++k)
    {
      j[k].reset(new int(k + 4));
    }
    cuda::std::swap(i, j);
    assert(*i[0] == 4);
    assert(*i[1] == 5);
    assert(*i[2] == 6);
    assert(*j[0] == 1);
    assert(*j[1] == 2);
    assert(*j[2] == 3);
  }
  {
    using CA = CopyOnly[42];
    using MA = NoexceptMoveOnly[42];
    using NA = NotMoveConstructible[42];
    static_assert(can_swap<CA&>(), "");
    static_assert(can_swap<MA&>(), "");
    static_assert(!can_swap<NA&>(), "");

    CA ca;
    MA ma;
    static_assert(!noexcept(cuda::std::swap(ca, ca)), "");
    static_assert(noexcept(cuda::std::swap(ma, ma)), "");
  }

  static_assert(test_swap_constexpr(), "");

  test_ambiguous_std();

  return 0;
}
