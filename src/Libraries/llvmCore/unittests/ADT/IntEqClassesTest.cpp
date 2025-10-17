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

//===---- ADT/IntEqClassesTest.cpp - IntEqClasses unit tests ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/IntEqClasses.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(IntEqClasses, Simple) {
  IntEqClasses ec(10);

  ec.join(0, 1);
  ec.join(3, 2);
  ec.join(4, 5);
  ec.join(7, 6);

  EXPECT_EQ(0u, ec.findLeader(0));
  EXPECT_EQ(0u, ec.findLeader(1));
  EXPECT_EQ(2u, ec.findLeader(2));
  EXPECT_EQ(2u, ec.findLeader(3));
  EXPECT_EQ(4u, ec.findLeader(4));
  EXPECT_EQ(4u, ec.findLeader(5));
  EXPECT_EQ(6u, ec.findLeader(6));
  EXPECT_EQ(6u, ec.findLeader(7));
  EXPECT_EQ(8u, ec.findLeader(8));
  EXPECT_EQ(9u, ec.findLeader(9));

  // join two non-leaders.
  ec.join(1, 3);

  EXPECT_EQ(0u, ec.findLeader(0));
  EXPECT_EQ(0u, ec.findLeader(1));
  EXPECT_EQ(0u, ec.findLeader(2));
  EXPECT_EQ(0u, ec.findLeader(3));
  EXPECT_EQ(4u, ec.findLeader(4));
  EXPECT_EQ(4u, ec.findLeader(5));
  EXPECT_EQ(6u, ec.findLeader(6));
  EXPECT_EQ(6u, ec.findLeader(7));
  EXPECT_EQ(8u, ec.findLeader(8));
  EXPECT_EQ(9u, ec.findLeader(9));

  // join two leaders.
  ec.join(4, 8);

  EXPECT_EQ(0u, ec.findLeader(0));
  EXPECT_EQ(0u, ec.findLeader(1));
  EXPECT_EQ(0u, ec.findLeader(2));
  EXPECT_EQ(0u, ec.findLeader(3));
  EXPECT_EQ(4u, ec.findLeader(4));
  EXPECT_EQ(4u, ec.findLeader(5));
  EXPECT_EQ(6u, ec.findLeader(6));
  EXPECT_EQ(6u, ec.findLeader(7));
  EXPECT_EQ(4u, ec.findLeader(8));
  EXPECT_EQ(9u, ec.findLeader(9));

  // join mixed.
  ec.join(9, 1);

  EXPECT_EQ(0u, ec.findLeader(0));
  EXPECT_EQ(0u, ec.findLeader(1));
  EXPECT_EQ(0u, ec.findLeader(2));
  EXPECT_EQ(0u, ec.findLeader(3));
  EXPECT_EQ(4u, ec.findLeader(4));
  EXPECT_EQ(4u, ec.findLeader(5));
  EXPECT_EQ(6u, ec.findLeader(6));
  EXPECT_EQ(6u, ec.findLeader(7));
  EXPECT_EQ(4u, ec.findLeader(8));
  EXPECT_EQ(0u, ec.findLeader(9));

  // compressed map.
  ec.compress();
  EXPECT_EQ(3u, ec.getNumClasses());

  EXPECT_EQ(0u, ec[0]);
  EXPECT_EQ(0u, ec[1]);
  EXPECT_EQ(0u, ec[2]);
  EXPECT_EQ(0u, ec[3]);
  EXPECT_EQ(1u, ec[4]);
  EXPECT_EQ(1u, ec[5]);
  EXPECT_EQ(2u, ec[6]);
  EXPECT_EQ(2u, ec[7]);
  EXPECT_EQ(1u, ec[8]);
  EXPECT_EQ(0u, ec[9]);

  // uncompressed map.
  ec.uncompress();
  EXPECT_EQ(0u, ec.findLeader(0));
  EXPECT_EQ(0u, ec.findLeader(1));
  EXPECT_EQ(0u, ec.findLeader(2));
  EXPECT_EQ(0u, ec.findLeader(3));
  EXPECT_EQ(4u, ec.findLeader(4));
  EXPECT_EQ(4u, ec.findLeader(5));
  EXPECT_EQ(6u, ec.findLeader(6));
  EXPECT_EQ(6u, ec.findLeader(7));
  EXPECT_EQ(4u, ec.findLeader(8));
  EXPECT_EQ(0u, ec.findLeader(9));
}

} // end anonymous namespace
