/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 25, 2025.
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

// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: clang
// XFAIL: *

// This tests is meant to demonstrate an existing ABI bug between the
// C++03 and C++11 implementations of cuda::std::function. It is not a real test.

// RUN: %cxx -c %s -o %t.first.o %flags %compile_flags -std=c++03 -g
// RUN: %cxx -c %s -o %t.second.o -DWITH_MAIN %flags %compile_flags -g -std=c++11
// RUN: %cxx -o %t.exe %t.first.o %t.second.o %flags %link_flags -g
// RUN: %run

#include <uscl/std/cassert>
#include <uscl/std/functional>

typedef cuda::std::function<void(int)> Func;

Func CreateFunc();

#ifndef WITH_MAIN
// In C++03, the functions call operator, which is a part of the vtable,
// is defined as 'void operator()(int)', but in C++11 it's
// void operator()(int&&)'. So when the C++03 version is passed to C++11 code
// the value of the integer is interpreted as its address.
void test(int x)
{
  assert(x == 42);
}
Func CreateFunc()
{
  Func f(&test);
  return f;
}
#else
int main()
{
  Func f = CreateFunc();
  f(42);
}
#endif
