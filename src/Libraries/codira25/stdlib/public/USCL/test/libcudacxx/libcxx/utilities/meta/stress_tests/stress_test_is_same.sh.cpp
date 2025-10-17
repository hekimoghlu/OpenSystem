/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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
//
// This is a dummy feature that prevents this test from running by default.
// REQUIRES: template-cost-testing

// The table below compares the compile time and object size for each of the
// variants listed in the RUN script.
//
//  Impl          Compile Time    Object Size
// -------------------------------------------
// cuda::std::_IsSame:    689.634 ms     356 K
// cuda::std::is_same:  8,129.180 ms     560 K
//
// RUN: %cxx %flags %compile_flags -c %s -o %S/orig.o -ggdb  -ggnu-pubnames -ftemplate-depth=5000 -ftime-trace
// -std=c++17 RUN: %cxx %flags %compile_flags -c %s -o %S/new.o -ggdb  -ggnu-pubnames -ftemplate-depth=5000 -ftime-trace
// -std=c++17 -DTEST_NEW

#include <uscl/std/cassert>
#include <uscl/std/type_traits>

#include "template_cost_testing.h"
#include "test_macros.h"

template <int N>
struct Arg
{
  enum
  {
    value = 1
  };
};

#ifdef TEST_NEW
#  define IS_SAME cuda::std::_IsSame
#else
#  define IS_SAME cuda::std::is_same
#endif

#define TEST_CASE_NOP()  IS_SAME<Arg<__COUNTER__>, Arg<__COUNTER__>>::value,
#define TEST_CASE_TYPE() IS_SAME<Arg<__COUNTER__>, Arg<__COUNTER__>>,

int sink(...);

int x = sink(REPEAT_10000(TEST_CASE_NOP) REPEAT_10000(TEST_CASE_NOP) 42);

void Foo(REPEAT_1000(TEST_CASE_TYPE) int) {}

static_assert(__COUNTER__ > 10000, "");

void escape()
{
  sink(&x);
  sink(&Foo);
}
