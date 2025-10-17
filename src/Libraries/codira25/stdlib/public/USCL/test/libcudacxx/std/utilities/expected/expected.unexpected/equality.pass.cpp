/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 17, 2022.
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

// template<class E2>
// friend constexpr bool operator==(const unexpected& x, const unexpected<E2>& y);
//
// Mandates: The expression x.error() == y.error() is well-formed and its result is convertible to bool.
//
// Returns: x.error() == y.error().

#include <uscl/std/cassert>
#include <uscl/std/concepts>
#include <uscl/std/expected>
#include <uscl/std/utility>

#include "test_macros.h"

struct Error
{
  int i;
#if TEST_STD_VER > 2017
  __host__ __device__ friend constexpr bool operator==(const Error&, const Error&) = default;
#else
  __host__ __device__ friend constexpr bool operator==(const Error& lhs, const Error& rhs) noexcept
  {
    return lhs.i == rhs.i;
  }
  __host__ __device__ friend constexpr bool operator!=(const Error& lhs, const Error& rhs) noexcept
  {
    return lhs.i != rhs.i;
  }
#endif
};

__host__ __device__ constexpr bool test()
{
  cuda::std::unexpected<Error> unex1(Error{2});
  cuda::std::unexpected<Error> unex2(Error{3});
  cuda::std::unexpected<Error> unex3(Error{2});
  assert(unex1 == unex3);
  assert(unex1 != unex2);
  assert(unex2 != unex3);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
