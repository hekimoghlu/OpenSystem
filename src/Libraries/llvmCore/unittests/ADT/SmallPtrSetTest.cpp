/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 7, 2024.
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

//===- llvm/unittest/ADT/SmallPtrSetTest.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// SmallPtrSet unit tests.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace llvm;

// SmallPtrSet swapping test.
TEST(SmallPtrSetTest, SwapTest) {
  int buf[10];

  SmallPtrSet<int *, 2> a;
  SmallPtrSet<int *, 2> b;

  a.insert(&buf[0]);
  a.insert(&buf[1]);
  b.insert(&buf[2]);

  std::swap(a, b);

  EXPECT_EQ(1U, a.size());
  EXPECT_EQ(2U, b.size());
  EXPECT_TRUE(a.count(&buf[2]));
  EXPECT_TRUE(b.count(&buf[0]));
  EXPECT_TRUE(b.count(&buf[1]));

  b.insert(&buf[3]);
  std::swap(a, b);

  EXPECT_EQ(3U, a.size());
  EXPECT_EQ(1U, b.size());
  EXPECT_TRUE(a.count(&buf[0]));
  EXPECT_TRUE(a.count(&buf[1]));
  EXPECT_TRUE(a.count(&buf[3]));
  EXPECT_TRUE(b.count(&buf[2]));

  std::swap(a, b);

  EXPECT_EQ(1U, a.size());
  EXPECT_EQ(3U, b.size());
  EXPECT_TRUE(a.count(&buf[2]));
  EXPECT_TRUE(b.count(&buf[0]));
  EXPECT_TRUE(b.count(&buf[1]));
  EXPECT_TRUE(b.count(&buf[3]));

  a.insert(&buf[4]);
  a.insert(&buf[5]);
  a.insert(&buf[6]);

  std::swap(b, a);

  EXPECT_EQ(3U, a.size());
  EXPECT_EQ(4U, b.size());
  EXPECT_TRUE(b.count(&buf[2]));
  EXPECT_TRUE(b.count(&buf[4]));
  EXPECT_TRUE(b.count(&buf[5]));
  EXPECT_TRUE(b.count(&buf[6]));
  EXPECT_TRUE(a.count(&buf[0]));
  EXPECT_TRUE(a.count(&buf[1]));
  EXPECT_TRUE(a.count(&buf[3]));
}
