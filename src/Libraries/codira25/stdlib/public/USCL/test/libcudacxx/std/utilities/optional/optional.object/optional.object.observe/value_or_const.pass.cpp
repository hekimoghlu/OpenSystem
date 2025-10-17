/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// template <class U> constexpr T optional<T>::value_or(U&& v) const&;

#include <uscl/std/cassert>
#include <uscl/std/optional>
#include <uscl/std/type_traits>

#include "test_macros.h"

using cuda::std::optional;

struct Y
{
  int i_;

  __host__ __device__ constexpr Y(int i)
      : i_(i)
  {}
};

struct X
{
  int i_;

  __host__ __device__ constexpr X(int i)
      : i_(i)
  {}
  __host__ __device__ constexpr X(const Y& y)
      : i_(y.i_)
  {}
  __host__ __device__ constexpr X(Y&& y)
      : i_(y.i_ + 1)
  {}
  __host__ __device__ friend constexpr bool operator==(const X& x, const X& y)
  {
    return x.i_ == y.i_;
  }
};

__host__ __device__ constexpr bool test()
{
  {
    const optional<X> opt(2);
    const Y y(3);
    assert(opt.value_or(y) == 2);
  }
  {
    const optional<X> opt(2);
    assert(opt.value_or(Y(3)) == 2);
  }
  {
    const optional<X> opt{};
    const Y y(3);
    assert(opt.value_or(y) == 3);
  }
  {
    const optional<X> opt{};
    assert(opt.value_or(Y(3)) == 4);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const optional<X>>().value_or(Y(3))), X>);
  }

  X val{2};
  {
    const optional<X&> opt(val);
    const Y y(3);
    assert(opt.value_or(y) == 2);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const optional<X&>>().value_or(y)), X>);
  }
  {
    const optional<X&> opt(val);
    assert(opt.value_or(Y(3)) == 2);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const optional<X&>>().value_or(Y(3))), X>);
  }
  {
    const optional<X&> opt{};
    const Y y(3);
    assert(opt.value_or(y) == 3);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const optional<X>>().value_or(y)), X>);
  }
  {
    const optional<X&> opt{};
    assert(opt.value_or(Y(3)) == 4);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const optional<X&>>().value_or(Y(3))), X>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
