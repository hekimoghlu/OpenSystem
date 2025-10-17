/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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
// UNSUPPORTED: c++17
// XFAIL: *

// <chrono>
// class year_month_weekday_last;

// template<class charT, class traits>
//     basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const year_month_weekday_last& ymwdl);
//
//   Returns: os << ymwdl.year() << '/' << ymwdl.month() << '/' << ymwdl.weekday_last().

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include <iostream>

#include "test_macros.h"

int main(int, char**)
{
  using year_month_weekday_last = cuda::std::chrono::year_month_weekday_last;
  using year                    = cuda::std::chrono::year;
  using month                   = cuda::std::chrono::month;
  using weekday                 = cuda::std::chrono::weekday;
  using weekday_last            = cuda::std::chrono::weekday_last;

  std::cout << year_month_weekday_last{year{2018}, month{3}, weekday_last{weekday{4}}};

  return 0;
}
