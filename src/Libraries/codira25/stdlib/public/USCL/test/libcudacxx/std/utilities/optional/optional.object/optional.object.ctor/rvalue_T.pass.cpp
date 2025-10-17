/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 10, 2024.
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
//

// <cuda/std/optional>

// constexpr optional(T&& v);

#include <uscl/std/cassert>
#include <uscl/std/optional>
#include <uscl/std/type_traits>

#include "archetypes.h"
#include "test_macros.h"

using cuda::std::optional;

template <class T>
__host__ __device__ constexpr void test()
{
  static_assert(cuda::std::is_convertible_v<const T&, optional<T>> == cuda::std::is_convertible_v<const T&, T>, "");
  {
    cuda::std::remove_reference_t<T> input{42};
    optional<T> opt{cuda::std::move(input)};
    assert(opt.has_value());
    assert(*opt == input);
  }
}

__host__ __device__ constexpr bool test()
{
  test<int>();
  test<double>();
  test<const int>();

  test<ConstexprTestTypes::TestType>();
  test<ExplicitConstexprTestTypes::TestType>();

  return true;
}

#if TEST_HAS_EXCEPTIONS()
class Z
{
public:
  Z(int) {}
  Z(Z&&)
  {
    TEST_THROW(6);
  }
};

void test_exceptions()
{
  try
  {
    Z z(3);
    optional<Z> opt(cuda::std::move(z));
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
  {
    typedef TestTypes::TestType T;
    T::reset();
    optional<T> opt = T{3};
    assert(T::alive() == 1);
    assert(T::move_constructed() == 1);
    assert(static_cast<bool>(opt) == true);
    assert(opt.value().value == 3);
  }
  {
    typedef ExplicitTestTypes::TestType T;
    static_assert(!cuda::std::is_convertible<T&&, optional<T>>::value, "");
    T::reset();
    optional<T> opt(T{3});
    assert(T::alive() == 1);
    assert(T::move_constructed() == 1);
    assert(static_cast<bool>(opt) == true);
    assert(opt.value().value == 3);
  }
  {
    typedef TestTypes::TestType T;
    T::reset();
    optional<T> opt = {3};
    assert(T::alive() == 1);
    assert(T::value_constructed() == 1);
    assert(T::copy_constructed() == 0);
    assert(T::move_constructed() == 0);
    assert(static_cast<bool>(opt) == true);
    assert(opt.value().value == 3);
  }
#if TEST_HAS_EXCEPTIONS()
  {
    NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
  }
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
