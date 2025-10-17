/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 1, 2024.
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

//===- llvm/unittest/ADT/FoldingSetTest.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// FoldingSet unit tests.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/FoldingSet.h"
#include <string>

using namespace llvm;

namespace {

// Unaligned string test.
TEST(FoldingSetTest, UnalignedStringTest) {
  SCOPED_TRACE("UnalignedStringTest");

  FoldingSetNodeID a, b;
  // An aligned string
  std::string str1= "a test string";
  a.AddString(str1);

  // An unaligned string
  std::string str2 = ">" + str1;
  b.AddString(str2.c_str() + 1);

  EXPECT_EQ(a.ComputeHash(), b.ComputeHash());
}

}

