/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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

#include <uscl/std/__memory_>
#include <uscl/std/string>
#include <uscl/std/tuple>

#include "test_macros.h"

struct UserType
{};

void test_bad_index()
{
  cuda::std::tuple<long, long, char, cuda::std::string, char, UserType, char> t1;
  TEST_IGNORE_NODISCARD cuda::std::get<int>(t1); // expected-error@tuple:* {{type not found}}
  TEST_IGNORE_NODISCARD cuda::std::get<long>(t1); // expected-note {{requested here}}
  TEST_IGNORE_NODISCARD cuda::std::get<char>(t1); // expected-note {{requested here}}
                                                  // expected-error@tuple:* 2 {{type occurs more than once}}
  cuda::std::tuple<> t0;
  TEST_IGNORE_NODISCARD cuda::std::get<char*>(t0); // expected-node {{requested here}}
                                                   // expected-error@tuple:* 1 {{type not in empty type list}}
}

void test_bad_return_type()
{
  using upint = cuda::std::unique_ptr<int>;
  cuda::std::tuple<upint> t;
  upint p = cuda::std::get<upint>(t); // expected-error{{deleted copy constructor}}
}

int main(int, char**)
{
  test_bad_index();
  test_bad_return_type();

  return 0;
}
