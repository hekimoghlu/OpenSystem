/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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

// XFAIL: *

// <chrono>
// class weekday_indexed;

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//   operator<<(basic_ostream<charT, traits>& os, const weekday_indexed& wdi);
//
//   Effects: os << wdi.weekday() << '[' << wdi.index().
//     If wdi.index() is in the range [1, 5], appends with ']',
//       otherwise appends with " is not a valid index]".

#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using weekday_indexed = cuda::std::chrono::weekday_indexed;
  using weekday         = cuda::std::chrono::weekday;

  std::cout << weekday_indexed{weekday{3}};

  return 0;
}
