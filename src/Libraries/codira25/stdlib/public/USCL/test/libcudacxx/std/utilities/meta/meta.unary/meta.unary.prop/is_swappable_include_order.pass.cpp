/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 23, 2022.
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
//
//===----------------------------------------------------------------------===//

// type_traits

// is_swappable

// If we're just building the test and not executing it, it should pass.
// UNSUPPORTED: no_execute

// IMPORTANT: The include order is part of the test. We want to pick up
// the following definitions in this order:
//   1) is_swappable, is_nothrow_swappable
//   2) iter_swap, swap_ranges
//   3) swap(T (&)[N], T (&)[N])
// This test checks that (1) and (2) see forward declarations
// for (3).
#include <uscl/std/type_traits>
// #include <uscl/std/algorithm>
#include <uscl/std/array>
#include <uscl/std/utility>

#include "test_macros.h"

int main(int, char**)
{
  // Use a builtin type so we don't get ADL lookup.
  typedef double T[17][29];
  {
    static_assert(cuda::std::__is_swappable<T>::value, "");
    static_assert(cuda::std::is_swappable_v<T>, "");
  }
  {
    T t1 = {};
    T t2 = {};
    cuda::std::iter_swap(t1, t2);
    cuda::std::swap_ranges(t1, t1 + 17, t2);
  }

  return 0;
}
