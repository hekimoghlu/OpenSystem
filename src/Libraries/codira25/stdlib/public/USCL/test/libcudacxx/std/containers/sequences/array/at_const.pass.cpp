/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 10, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// const_reference at (size_type) const; // constexpr in C++14

#include <uscl/std/array>
#include <uscl/std/cassert>

#include "test_macros.h"

#if TEST_HAS_EXCEPTIONS()
#  include <stdexcept>
#endif // TEST_HAS_EXCEPTIONS()

__host__ __device__ constexpr bool tests()
{
  {
    typedef double T;
    typedef cuda::std::array<T, 3> C;
    C const c                      = {1, 2, 3.5};
    typename C::const_reference r1 = c.at(0);
    assert(r1 == 1);

    typename C::const_reference r2 = c.at(2);
    assert(r2 == 3.5);
  }
  return true;
}

#if TEST_HAS_EXCEPTIONS()
void test_exceptions()
{
  {
    cuda::std::array<int, 4> const array = {1, 2, 3, 4};

    try
    {
      TEST_IGNORE_NODISCARD array.at(4);
      assert(false);
    }
    catch (std::out_of_range const&)
    {
      // pass
    }
    catch (...)
    {
      assert(false);
    }

    try
    {
      TEST_IGNORE_NODISCARD array.at(5);
      assert(false);
    }
    catch (std::out_of_range const&)
    {
      // pass
    }
    catch (...)
    {
      assert(false);
    }

    try
    {
      TEST_IGNORE_NODISCARD array.at(6);
      assert(false);
    }
    catch (std::out_of_range const&)
    {
      // pass
    }
    catch (...)
    {
      assert(false);
    }

    try
    {
      using size_type = decltype(array)::size_type;
      TEST_IGNORE_NODISCARD array.at(static_cast<size_type>(-1));
      assert(false);
    }
    catch (std::out_of_range const&)
    {
      // pass
    }
    catch (...)
    {
      assert(false);
    }
  }

  {
    cuda::std::array<int, 0> array = {};

    try
    {
      TEST_IGNORE_NODISCARD array.at(0);
      assert(false);
    }
    catch (std::out_of_range const&)
    {
      // pass
    }
    catch (...)
    {
      assert(false);
    }
  }
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  tests();
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()

  static_assert(tests(), "");
  return 0;
}
