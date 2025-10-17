/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 27, 2024.
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

// constexpr optional(const optional<T>& rhs);

#include <uscl/std/cassert>
#include <uscl/std/optional>
#include <uscl/std/type_traits>

#include "archetypes.h"
#include "test_macros.h"

using cuda::std::optional;

template <class T>
__host__ __device__ constexpr void test() noexcept
{
  { // copy constructed from empty
    const optional<T> input{};
    optional<T> opt{input};
    assert(!input.has_value());
    assert(!opt.has_value());
  }

  { // copy constructed from empty
    cuda::std::remove_reference_t<T> val{42};
    const optional<T> input{val};
    optional<T> opt{input};
    assert(input.has_value());
    assert(opt.has_value());
    assert(*opt == val);
    if constexpr (cuda::std::is_reference_v<T>)
    {
      assert(cuda::std::addressof(val) == opt.operator->());
    }
  }
}

__host__ __device__ constexpr bool test() noexcept
{
  test<int>();
  test<const int>();
  test<ConstexprTestTypes::TestType>();
  test<TrivialTestTypes::TestType>();

#ifdef CCCL_ENABLE_OPTIONAL_REF
  test<int&>();
#endif // CCCL_ENABLE_OPTIONAL_REF

  return true;
}

#if TEST_HAS_EXCEPTIONS()
struct Z
{
  Z()
      : count(0)
  {}
  Z(Z const& o)
      : count(o.count + 1)
  {
    if (count == 2)
    {
      TEST_THROW(6);
    }
  }
  int count;
};

void test_throwing_ctor()
{
  const Z z;
  const optional<Z> rhs(z);
  try
  {
    optional<Z> lhs(rhs);
    assert(false);
  }
  catch (int i)
  {
    assert(i == 6);
  }
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)

  {
    using T = TestTypes::TestType;
    T::reset();
    const optional<T> rhs;
    assert(T::alive() == 0);
    const optional<T> lhs(rhs);
    assert(lhs.has_value() == false);
    assert(T::alive() == 0);
  }
  TestTypes::TestType::reset();
  {
    using T = TestTypes::TestType;
    T::reset();
    const optional<T> rhs(42);
    assert(T::alive() == 1);
    assert(T::value_constructed() == 1);
    assert(T::copy_constructed() == 0);
    const optional<T> lhs(rhs);
    assert(lhs.has_value());
    assert(T::copy_constructed() == 1);
    assert(T::alive() == 2);
  }

#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_throwing_ctor();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
