/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 7, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16

// gcc is unable to get the construction of b right
// UNSUPPORTED: gcc-7, gcc-8, gcc-9

// class cuda::std::ranges::subrange;
//   Test the example from LWG 3470,
//   qualification conversions in __convertible_to_non_slicing

#include <uscl/std/cassert>
#include <uscl/std/ranges>

#include "test_macros.h"

using gcc_needs_help_type = cuda::std::ranges::subrange<int**>;

__host__ __device__ constexpr bool test()
{
  // The example from LWG3470, using implicit conversion.
  int a[3]                                         = {1, 2, 3};
  int* b[3]                                        = {&a[2], &a[0], &a[1]};
  cuda::std::ranges::subrange<const int* const*> c = b;
  assert(c.begin() == b + 0);
  assert(c.end() == b + 3);

  // Also test CTAD and a subrange-to-subrange conversion.
  cuda::std::ranges::subrange d{b};
  static_assert(cuda::std::same_as<decltype(d), gcc_needs_help_type>);
  assert(d.begin() == b + 0);
  assert(d.end() == b + 3);

  cuda::std::ranges::subrange<const int* const*> e = d;
  assert(e.begin() == b + 0);
  assert(e.end() == b + 3);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
