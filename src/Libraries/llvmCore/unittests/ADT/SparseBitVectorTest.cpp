/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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

//===- llvm/unittest/ADT/SparseBitVectorTest.cpp - SparseBitVector tests --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SparseBitVector.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(SparseBitVectorTest, TrivialOperation) {
  SparseBitVector<> Vec;
  EXPECT_EQ(0U, Vec.count());
  EXPECT_FALSE(Vec.test(17));
  Vec.set(5);
  EXPECT_TRUE(Vec.test(5));
  EXPECT_FALSE(Vec.test(17));
  Vec.reset(6);
  EXPECT_TRUE(Vec.test(5));
  EXPECT_FALSE(Vec.test(6));
  Vec.reset(5);
  EXPECT_FALSE(Vec.test(5));
  EXPECT_TRUE(Vec.test_and_set(17));
  EXPECT_FALSE(Vec.test_and_set(17));
  EXPECT_TRUE(Vec.test(17));
  Vec.clear();
  EXPECT_FALSE(Vec.test(17));
}

}
