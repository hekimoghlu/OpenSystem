/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 3, 2024.
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

// constexpr void value() const &;
// constexpr void value() &&;

#include <uscl/std/cassert>
#include <uscl/std/concepts>
#include <uscl/std/expected>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  // const &
  {
    const cuda::std::expected<void, int> e;
    e.value();
    static_assert(cuda::std::is_same_v<decltype(e.value()), void>, "");
  }

  // &
  {
    cuda::std::expected<void, int> e;
    e.value();
    static_assert(cuda::std::is_same_v<decltype(e.value()), void>, "");
  }

  // &&
  {
    cuda::std::expected<void, int> e;
    cuda::std::move(e).value();
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(e).value()), void>, "");
  }

  // const &&
  {
    const cuda::std::expected<void, int> e;
    cuda::std::move(e).value();
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(e).value()), void>, "");
  }

  return true;
}

#if TEST_HAS_EXCEPTIONS()
void test_exceptions()
{
  // Test const& overload
  try
  {
    const cuda::std::expected<void, int> e(cuda::std::unexpect, 5);
    e.value();
    assert(false);
  }
  catch (const cuda::std::bad_expected_access<int>& ex)
  {
    assert(ex.error() == 5);
  }
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
  static_assert(test(), "");

#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()
  return 0;
}
