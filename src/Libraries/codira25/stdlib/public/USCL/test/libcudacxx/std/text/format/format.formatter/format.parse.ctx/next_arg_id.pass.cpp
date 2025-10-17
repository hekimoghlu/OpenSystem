/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 30, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// constexpr size_t next_arg_id();

#include <uscl/std/__format_>
#include <uscl/std/cassert>
#include <uscl/std/string_view>

#include "literal.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  cuda::std::format_parse_context context("", 10);
  for (cuda::std::size_t i = 0; i < 10; ++i)
  {
    assert(i == context.next_arg_id());
  }

  return true;
}

#if TEST_HAS_EXCEPTIONS()
void test_exception()
{
  cuda::std::format_parse_context context("", 1);
  context.check_arg_id(0);

  try
  {
    (void) context.next_arg_id();
    assert(false);
  }
  catch (const cuda::std::format_error&)
  {}
  catch (...)
  {
    assert(false);
  }
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
  static_assert(test());
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exception();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
