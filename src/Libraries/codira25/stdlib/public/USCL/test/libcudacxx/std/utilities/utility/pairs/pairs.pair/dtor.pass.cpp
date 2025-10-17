/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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

// <utility>

// template <class T1, class T2> struct pair

// ~pair()

// C++17 added:
//   The destructor of pair shall be a trivial destructor
//     if (is_trivially_destructible_v<T1> && is_trivially_destructible_v<T2>) is true.

#include <uscl/std/type_traits>
#include <uscl/std/utility>
// cuda::std::string not supported
// #include <uscl/std/string>
#include <uscl/std/cassert>

#include "DefaultOnly.h"
#include "test_macros.h"

int main(int, char**)
{
  static_assert((cuda::std::is_trivially_destructible<cuda::std::pair<int, float>>::value), "");
  /*
  static_assert((!cuda::std::is_trivially_destructible<
      cuda::std::pair<int, cuda::std::string> >::value), "");
  */
  static_assert((!cuda::std::is_trivially_destructible<cuda::std::pair<int, DefaultOnly>>::value), "");

  return 0;
}
