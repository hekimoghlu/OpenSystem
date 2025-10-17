/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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

// template <class U>
//   explicit optional(optional<U>&& rhs);

#include <uscl/std/cassert>
#include <uscl/std/optional>
#include <uscl/std/type_traits>

#include "test_macros.h"

using cuda::std::optional;

class X
{
  int i_;

public:
  __host__ __device__ constexpr explicit X(int i)
      : i_(i)
  {}
  __host__ __device__ constexpr X(X&& x)
      : i_(x.i_)
  {
    x.i_ = 0;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX20 ~X()
  {
    i_ = 0;
  }
  __host__ __device__ friend constexpr bool operator==(const X& x, const int& y)
  {
    return x.i_ == y;
  }
};

#ifdef CCCL_ENABLE_OPTIONAL_REF
template <class T>
struct ConvertibleToReference
{
  T val_;

  __host__ __device__ constexpr operator T&() noexcept
  {
    return val_;
  }

  __host__ __device__ friend constexpr bool operator==(const int& lhs, const ConvertibleToReference& rhs) noexcept
  {
    return lhs == rhs.val_;
  }
};

template <class T>
struct ExplicitlyConvertibleToReference
{
  T val_;

  __host__ __device__ explicit constexpr operator T&() noexcept
  {
    return val_;
  }

  __host__ __device__ friend constexpr bool
  operator==(const int& lhs, const ExplicitlyConvertibleToReference& rhs) noexcept
  {
    return lhs == rhs.val_;
  }
};
#endif // CCCL_ENABLE_OPTIONAL_REF

template <class T, class U>
__host__ __device__ constexpr void test()
{
  { // constructed from empty
    optional<U> input{};
    optional<T> opt{cuda::std::move(input)};
    assert(!input.has_value());
    assert(!opt.has_value());
  }
  { // constructed from non-empty
    cuda::std::remove_reference_t<U> val{42};
    optional<U> input{val};
    optional<T> opt{cuda::std::move(input)};
    assert(input.has_value());
    assert(opt.has_value());
    assert(*opt == val);
    if constexpr (cuda::std::is_reference_v<T>)
    {
      // optional<U> does not necessarily hold a reference so we cannot use addressof(val)
      assert(cuda::std::addressof(static_cast<T>(*input)) == opt.operator->());
    }
  }
}

__host__ __device__ constexpr bool test()
{
  test<X, int>();

#ifdef CCCL_ENABLE_OPTIONAL_REF
  test<int&, ConvertibleToReference<int>>();
  test<int&, ExplicitlyConvertibleToReference<int>>();

  test<const int&, ConvertibleToReference<int>>();
  test<const int&, ExplicitlyConvertibleToReference<int>>();
#endif // CCCL_ENABLE_OPTIONAL_REF

  return true;
}

struct TerminatesOnConstruction
{
  __host__ __device__ explicit TerminatesOnConstruction(int)
  {
    cuda::std::terminate();
  }
};

#if TEST_HAS_EXCEPTIONS()
class Z
{
public:
  explicit Z(int)
  {
    TEST_THROW(6);
  }
};

template <class T, class U>
void test_exception(optional<U>&& rhs)
{
  static_assert(!(cuda::std::is_convertible<optional<U>&&, optional<T>>::value), "");
  try
  {
    optional<T> lhs(cuda::std::move(rhs));
    unused(lhs);
    assert(false);
  }
  catch (int i)
  {
    assert(i == 6);
  }
}

void test_exceptions()
{
  optional<int> rhs(3);
  test_exception<Z>(cuda::std::move(rhs));
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)

  {
    using T = TerminatesOnConstruction;
    optional<int> input{};
    optional<T> lhs{cuda::std::move(input)};
    assert(!input.has_value());
    assert(!lhs.has_value());
  }

#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
