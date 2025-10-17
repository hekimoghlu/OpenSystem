/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
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

//===--- SILBitfieldTest.cpp ----------------------------------------------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"


namespace language {

class BasicBlockBitfield;

// Fake SILFunction and SILBasicBlock.

struct SILFunction {
  BasicBlockBitfield *newestAliveBlockBitfield = nullptr;
  uint64_t currentBitfieldID = 1;
};

struct SILBasicBlock {
  SILFunction *function;
  uint32_t customBits = 0;
  uint64_t lastInitializedBitfieldID = 0;

  enum { numCustomBits = 32 };
  enum { maxBitfieldID = std::numeric_limits<uint64_t>::max() };

  SILBasicBlock(SILFunction *function): function(function) {}

  SILFunction *getFunction() const { return function; }
  
  unsigned getCustomBits() const { return customBits; }
  void setCustomBits(unsigned value) { customBits = value; }
  
};

}

#define LANGUAGE_SIL_SILFUNCTION_H

#include "language/SIL/BasicBlockBits.h"

using namespace language;

namespace {

TEST(SILBitfieldTest, Basic) {
  SILFunction f;
  SILBasicBlock b(&f);

  {
    BasicBlockFlag A(&f);
    EXPECT_FALSE(A.get(&b));

    EXPECT_FALSE(A.testAndSet(&b));
    EXPECT_TRUE(A.get(&b));

    EXPECT_TRUE(A.testAndSet(&b));
    EXPECT_TRUE(A.get(&b));

    A.reset(&b);
    EXPECT_FALSE(A.get(&b));

    {
      BasicBlockBitfield B(&f, 5);
      EXPECT_EQ(B.get(&b), 0u);

      B.set(&b, 27);
      EXPECT_FALSE(A.get(&b));
      EXPECT_EQ(B.get(&b), 27u);

      A.set(&b);
      EXPECT_TRUE(A.get(&b));
      EXPECT_EQ(B.get(&b), 27u);

      B.set(&b, 2);
      EXPECT_TRUE(A.get(&b));
      EXPECT_EQ(B.get(&b), 2u);

      B.set(&b, 31);
      EXPECT_TRUE(A.get(&b));
      EXPECT_EQ(B.get(&b), 31u);
    }
    {
      BasicBlockBitfield C(&f, 2);
      EXPECT_EQ(C.get(&b), 0u);

      BasicBlockFlag D(&f);
      EXPECT_FALSE(D.get(&b));

      BasicBlockBitfield E(&f, 3);

      E.set(&b, 5);
      EXPECT_TRUE(A.get(&b));
      EXPECT_EQ(C.get(&b), 0u);
      EXPECT_FALSE(D.get(&b));
      EXPECT_EQ(E.get(&b), 5u);

      C.set(&b, 1);
      EXPECT_TRUE(A.get(&b));
      EXPECT_EQ(C.get(&b), 1u);
      EXPECT_FALSE(D.get(&b));
      EXPECT_EQ(E.get(&b), 5u);
    }
  }
  {
    BasicBlockBitfield F(&f, 32);
    EXPECT_EQ(F.get(&b), 0u);
    F.set(&b, 0xdeadbeaf);
    EXPECT_EQ(F.get(&b), 0xdeadbeaf);
  }
}

}
