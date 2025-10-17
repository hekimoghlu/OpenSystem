/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr const T& value() const &;
// constexpr T& value() &;
// constexpr T&& value() &&;
// constexpr const T&& value() const &&;

#include <uscl/std/cassert>
#include <uscl/std/concepts>
#include <uscl/std/expected>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  // non-const &
  {
    cuda::std::expected<int, int> e(5);
    decltype(auto) x = e.value();
    static_assert(cuda::std::same_as<decltype(x), int&>, "");
    assert(&x == &(*e));
    assert(x == 5);
  }

  // const &
  {
    const cuda::std::expected<int, int> e(5);
    decltype(auto) x = e.value();
    static_assert(cuda::std::same_as<decltype(x), const int&>, "");
    assert(&x == &(*e));
    assert(x == 5);
  }

  // non-const &&
  {
    cuda::std::expected<int, int> e(5);
    decltype(auto) x = cuda::std::move(e).value();
    static_assert(cuda::std::same_as<decltype(x), int&&>, "");
    assert(&x == &(*e));
    assert(x == 5);
  }

  // const &&
  {
    const cuda::std::expected<int, int> e(5);
    decltype(auto) x = cuda::std::move(e).value();
    static_assert(cuda::std::same_as<decltype(x), const int&&>, "");
    assert(&x == &(*e));
    assert(x == 5);
  }

  return true;
}

#if TEST_HAS_EXCEPTIONS()
struct Error
{
  enum
  {
    Default,
    MutableRefCalled,
    ConstRefCalled,
    MutableRvalueCalled,
    ConstRvalueCalled
  } From  = Default;
  Error() = default;
  Error(const Error& e)
      : From(e.From)
  {
    if (e.From == Default)
    {
      From = ConstRefCalled;
    }
  }
  Error(Error& e)
      : From(e.From)
  {
    if (e.From == Default)
    {
      From = MutableRefCalled;
    }
  }
  Error(const Error&& e)
      : From(e.From)
  {
    if (e.From == Default)
    {
      From = ConstRvalueCalled;
    }
  }
  Error(Error&& e)
      : From(e.From)
  {
    if (e.From == Default)
    {
      From = MutableRvalueCalled;
    }
  }
};

void test_exceptions()
{
  try
  {
    const cuda::std::expected<int, int> e(cuda::std::unexpect, 5);
    (void) e.value();
    assert(false);
  }
  catch (const cuda::std::bad_expected_access<int>& ex)
  {
    assert(ex.error() == 5);
  }

  // Test & overload
  try
  {
    cuda::std::expected<int, Error> e(cuda::std::unexpect);
    (void) e.value();
    assert(false);
  }
  catch (const cuda::std::bad_expected_access<Error>& ex)
  {
    assert(ex.error().From == Error::ConstRefCalled);
  }

  // Test const& overload
  try
  {
    const cuda::std::expected<int, Error> e(cuda::std::unexpect);
    (void) e.value();
    assert(false);
  }
  catch (const cuda::std::bad_expected_access<Error>& ex)
  {
    assert(ex.error().From == Error::ConstRefCalled);
  }

  // Test && overload
  try
  {
    cuda::std::expected<int, Error> e(cuda::std::unexpect);
    (void) cuda::std::move(e).value();
    assert(false);
  }
  catch (const cuda::std::bad_expected_access<Error>& ex)
  {
    assert(ex.error().From == Error::MutableRvalueCalled);
  }

  // Test const&& overload
  try
  {
    const cuda::std::expected<int, Error> e(cuda::std::unexpect);
    (void) cuda::std::move(e).value();
    assert(false);
  }
  catch (const cuda::std::bad_expected_access<Error>& ex)
  {
    assert(ex.error().From == Error::ConstRvalueCalled);
  }
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()
  return 0;
}
