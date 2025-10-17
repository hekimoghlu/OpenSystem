/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 19, 2025.
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
#include "video/corruption_detection/halton_sequence.h"

#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

using ::testing::DoubleEq;
using ::testing::ElementsAre;

TEST(HaltonSequenceTest, ShouldGenerateBase2SequenceByDefault) {
  HaltonSequence halton_sequence;
  EXPECT_THAT(halton_sequence.GetNext(), ElementsAre(DoubleEq(0.0)));
  EXPECT_THAT(halton_sequence.GetNext(), ElementsAre(DoubleEq(1.0 / 2)));
  EXPECT_THAT(halton_sequence.GetNext(), ElementsAre(DoubleEq(1.0 / 4)));
  EXPECT_THAT(halton_sequence.GetNext(), ElementsAre(DoubleEq(3.0 / 4)));
  EXPECT_THAT(halton_sequence.GetNext(), ElementsAre(DoubleEq(1.0 / 8)));
  EXPECT_THAT(halton_sequence.GetNext(), ElementsAre(DoubleEq(5.0 / 8)));
  EXPECT_THAT(halton_sequence.GetNext(), ElementsAre(DoubleEq(3.0 / 8)));
}

TEST(HaltonSequenceTest,
     ShouldGenerateBase2Base3SequencesWhenCreatedAs2Dimensional) {
  HaltonSequence halton_sequence(2);
  EXPECT_THAT(halton_sequence.GetNext(),
              ElementsAre(DoubleEq(0.0), DoubleEq(0.0)));
  EXPECT_THAT(halton_sequence.GetNext(),
              ElementsAre(DoubleEq(1.0 / 2), DoubleEq(1.0 / 3)));
  EXPECT_THAT(halton_sequence.GetNext(),
              ElementsAre(DoubleEq(1.0 / 4), DoubleEq(2.0 / 3)));
  EXPECT_THAT(halton_sequence.GetNext(),
              ElementsAre(DoubleEq(3.0 / 4), DoubleEq(1.0 / 9)));
  EXPECT_THAT(halton_sequence.GetNext(),
              ElementsAre(DoubleEq(1.0 / 8), DoubleEq(4.0 / 9)));
  EXPECT_THAT(halton_sequence.GetNext(),
              ElementsAre(DoubleEq(5.0 / 8), DoubleEq(7.0 / 9)));
  EXPECT_THAT(halton_sequence.GetNext(),
              ElementsAre(DoubleEq(3.0 / 8), DoubleEq(2.0 / 9)));
}

TEST(HaltonSequenceTest, ShouldRestartSequenceWhenResetIsCalled) {
  HaltonSequence halton_sequence;
  EXPECT_THAT(halton_sequence.GetCurrentIndex(), 0);
  EXPECT_THAT(halton_sequence.GetNext(), ElementsAre(DoubleEq(0.0)));
  EXPECT_THAT(halton_sequence.GetCurrentIndex(), 1);
  EXPECT_THAT(halton_sequence.GetNext(), ElementsAre(DoubleEq(1.0 / 2)));
  EXPECT_THAT(halton_sequence.GetCurrentIndex(), 2);
  halton_sequence.Reset();
  EXPECT_THAT(halton_sequence.GetCurrentIndex(), 0);
  EXPECT_THAT(halton_sequence.GetNext(), ElementsAre(DoubleEq(0.0)));
}

TEST(HaltonSequenceTest, ShouldSetCurrentIndexWhenSetCurrentIndexIsCalled) {
  HaltonSequence halton_sequence;
  EXPECT_THAT(halton_sequence.GetCurrentIndex(), 0);
  halton_sequence.SetCurrentIndex(3);
  EXPECT_THAT(halton_sequence.GetCurrentIndex(), 3);
  EXPECT_THAT(halton_sequence.GetNext(), ElementsAre(DoubleEq(3.0 / 4)));
}

}  // namespace
}  // namespace webrtc
