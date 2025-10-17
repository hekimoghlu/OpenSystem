/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
// class year;

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//   operator<<(basic_ostream<charT, traits>& os, const year& y);
//
//   Effects: Inserts format(fmt, y) where fmt is "%Y" widened to charT.
//   If !y.ok(), appends with " is not a valid year".
//
// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//   to_stream(basic_ostream<charT, traits>& os, const charT* fmt, const year& y);
//
//   Effects: Streams y into os using the format specified by the NTCTS fmt.
//     fmt encoding follows the rules specified in 25.11.
//
// template<class charT, class traits, class Alloc = allocator<charT>>
//   basic_istream<charT, traits>&
//   from_stream(basic_istream<charT, traits>& is, const charT* fmt,
//               year& y, basic_string<charT, traits, Alloc>* abbrev = nullptr,
//               minutes* offset = nullptr);
//
//   Effects: Attempts to parse the input stream is into the year y using the format flags
//     given in the NTCTS fmt as specified in 25.12. If the parse fails to decode a valid year,
//     is.setstate(ios_base::failbit) shall be called and y shall not be modified. If %Z is used
//     and successfully parsed, that value will be assigned to *abbrev if abbrev is non-null.
//     If %z (or a modified variant) is used and successfully parsed, that value will be
//     assigned to *offset if offset is non-null.

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include <iostream>

#include "test_macros.h"

int main(int, char**)
{
  using year = cuda::std::chrono::year;

  std::cout << year{2018};

  return 0;
}
