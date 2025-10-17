/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 23, 2022.
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
#include "modules/video_coding/chain_diff_calculator.h"

#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

using ::testing::ElementsAre;
using ::testing::SizeIs;

TEST(ChainDiffCalculatorTest, SingleChain) {
  // Simulate  a stream with 2 temporal layer where chain
  // protects temporal layer 0.
  ChainDiffCalculator calculator;
  // Key frame.
  calculator.Reset({true});
  EXPECT_THAT(calculator.From(1, {true}), ElementsAre(0));
  // T1 delta frame.
  EXPECT_THAT(calculator.From(2, {false}), ElementsAre(1));
  // T0 delta frame.
  EXPECT_THAT(calculator.From(3, {true}), ElementsAre(2));
}

TEST(ChainDiffCalculatorTest, TwoChainsFullSvc) {
  // Simulate a full svc stream with 2 spatial and 2 temporal layers.
  // chains are protecting temporal layers 0.
  ChainDiffCalculator calculator;
  // S0 Key frame.
  calculator.Reset({true, true});
  EXPECT_THAT(calculator.From(1, {true, true}), ElementsAre(0, 0));
  // S1 Key frame.
  EXPECT_THAT(calculator.From(2, {false, true}), ElementsAre(1, 1));
  // S0T1 delta frame.
  EXPECT_THAT(calculator.From(3, {false, false}), ElementsAre(2, 1));
  // S1T1 delta frame.
  EXPECT_THAT(calculator.From(4, {false, false}), ElementsAre(3, 2));
  // S0T0 delta frame.
  EXPECT_THAT(calculator.From(5, {true, true}), ElementsAre(4, 3));
  // S1T0 delta frame.
  EXPECT_THAT(calculator.From(6, {false, true}), ElementsAre(1, 1));
}

TEST(ChainDiffCalculatorTest, TwoChainsKSvc) {
  // Simulate a k-svc stream with 2 spatial and 2 temporal layers.
  // chains are protecting temporal layers 0.
  ChainDiffCalculator calculator;
  // S0 Key frame.
  calculator.Reset({true, true});
  EXPECT_THAT(calculator.From(1, {true, true}), ElementsAre(0, 0));
  // S1 Key frame.
  EXPECT_THAT(calculator.From(2, {false, true}), ElementsAre(1, 1));
  // S0T1 delta frame.
  EXPECT_THAT(calculator.From(3, {false, false}), ElementsAre(2, 1));
  // S1T1 delta frame.
  EXPECT_THAT(calculator.From(4, {false, false}), ElementsAre(3, 2));
  // S0T0 delta frame.
  EXPECT_THAT(calculator.From(5, {true, false}), ElementsAre(4, 3));
  // S1T0 delta frame.
  EXPECT_THAT(calculator.From(6, {false, true}), ElementsAre(1, 4));
}

TEST(ChainDiffCalculatorTest, TwoChainsSimulcast) {
  // Simulate a k-svc stream with 2 spatial and 2 temporal layers.
  // chains are protecting temporal layers 0.
  ChainDiffCalculator calculator;
  // S0 Key frame.
  calculator.Reset({true, false});
  EXPECT_THAT(calculator.From(1, {true, false}), ElementsAre(0, 0));
  // S1 Key frame.
  calculator.Reset({false, true});
  EXPECT_THAT(calculator.From(2, {false, true}), ElementsAre(1, 0));
  // S0T1 delta frame.
  EXPECT_THAT(calculator.From(3, {false, false}), ElementsAre(2, 1));
  // S1T1 delta frame.
  EXPECT_THAT(calculator.From(4, {false, false}), ElementsAre(3, 2));
  // S0T0 delta frame.
  EXPECT_THAT(calculator.From(5, {true, false}), ElementsAre(4, 3));
  // S1T0 delta frame.
  EXPECT_THAT(calculator.From(6, {false, true}), ElementsAre(1, 4));
}

TEST(ChainDiffCalculatorTest, ResilentToAbsentChainConfig) {
  ChainDiffCalculator calculator;
  // Key frame.
  calculator.Reset({true, false});
  EXPECT_THAT(calculator.From(1, {true, false}), ElementsAre(0, 0));
  // Forgot to set chains. should still return 2 chain_diffs.
  EXPECT_THAT(calculator.From(2, {}), ElementsAre(1, 0));
  // chain diffs for next frame(s) are undefined, but still there should be
  // correct number of them.
  EXPECT_THAT(calculator.From(3, {true, false}), SizeIs(2));
  EXPECT_THAT(calculator.From(4, {false, true}), SizeIs(2));
  // Since previous two frames updated all the chains, can expect what
  // chain_diffs would be.
  EXPECT_THAT(calculator.From(5, {false, false}), ElementsAre(2, 1));
}

TEST(ChainDiffCalculatorTest, ResilentToTooMainChains) {
  ChainDiffCalculator calculator;
  // Key frame.
  calculator.Reset({true, false});
  EXPECT_THAT(calculator.From(1, {true, false}), ElementsAre(0, 0));
  // Set wrong number of chains. Expect number of chain_diffs is not changed.
  EXPECT_THAT(calculator.From(2, {true, true, true}), ElementsAre(1, 0));
  // chain diffs for next frame(s) are undefined, but still there should be
  // correct number of them.
  EXPECT_THAT(calculator.From(3, {true, false}), SizeIs(2));
  EXPECT_THAT(calculator.From(4, {false, true}), SizeIs(2));
  // Since previous two frames updated all the chains, can expect what
  // chain_diffs would be.
  EXPECT_THAT(calculator.From(5, {false, false}), ElementsAre(2, 1));
}

}  // namespace
}  // namespace webrtc
