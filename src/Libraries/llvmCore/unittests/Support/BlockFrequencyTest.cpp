/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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

#include "llvm/Support/DataTypes.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/BranchProbability.h"

#include "gtest/gtest.h"
#include <climits>

using namespace llvm;

namespace {

TEST(BlockFrequencyTest, OneToZero) {
  BlockFrequency Freq(1);
  BranchProbability Prob(UINT32_MAX - 1, UINT32_MAX);
  Freq *= Prob;
  EXPECT_EQ(Freq.getFrequency(), 0u);
}

TEST(BlockFrequencyTest, OneToOne) {
  BlockFrequency Freq(1);
  BranchProbability Prob(UINT32_MAX, UINT32_MAX);
  Freq *= Prob;
  EXPECT_EQ(Freq.getFrequency(), 1u);
}

TEST(BlockFrequencyTest, ThreeToOne) {
  BlockFrequency Freq(3);
  BranchProbability Prob(3000000, 9000000);
  Freq *= Prob;
  EXPECT_EQ(Freq.getFrequency(), 1u);
}

TEST(BlockFrequencyTest, MaxToHalfMax) {
  BlockFrequency Freq(UINT64_MAX);
  BranchProbability Prob(UINT32_MAX / 2, UINT32_MAX);
  Freq *= Prob;
  EXPECT_EQ(Freq.getFrequency(), 9223372034707292159ULL);
}

TEST(BlockFrequencyTest, BigToBig) {
  const uint64_t Big = 387246523487234346LL;
  const uint32_t P = 123456789;
  BlockFrequency Freq(Big);
  BranchProbability Prob(P, P);
  Freq *= Prob;
  EXPECT_EQ(Freq.getFrequency(), Big);
}

TEST(BlockFrequencyTest, MaxToMax) {
  BlockFrequency Freq(UINT64_MAX);
  BranchProbability Prob(UINT32_MAX, UINT32_MAX);
  Freq *= Prob;
  EXPECT_EQ(Freq.getFrequency(), UINT64_MAX);
}

TEST(BlockFrequencyTest, ProbabilityCompare) {
  BranchProbability A(4, 5);
  BranchProbability B(4U << 29, 5U << 29);
  BranchProbability C(3, 4);

  EXPECT_TRUE(A == B);
  EXPECT_FALSE(A != B);
  EXPECT_FALSE(A < B);
  EXPECT_FALSE(A > B);
  EXPECT_TRUE(A <= B);
  EXPECT_TRUE(A >= B);

  EXPECT_FALSE(B == C);
  EXPECT_TRUE(B != C);
  EXPECT_FALSE(B < C);
  EXPECT_TRUE(B > C);
  EXPECT_FALSE(B <= C);
  EXPECT_TRUE(B >= C);

  BranchProbability BigZero(0, UINT32_MAX);
  BranchProbability BigOne(UINT32_MAX, UINT32_MAX);
  EXPECT_FALSE(BigZero == BigOne);
  EXPECT_TRUE(BigZero != BigOne);
  EXPECT_TRUE(BigZero < BigOne);
  EXPECT_FALSE(BigZero > BigOne);
  EXPECT_TRUE(BigZero <= BigOne);
  EXPECT_FALSE(BigZero >= BigOne);
}

}
