/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 11, 2022.
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
#include "modules/video_coding/svc/scalability_structure_full_svc.h"

#include <vector>

#include "modules/video_coding/svc/scalability_structure_test_helpers.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

using ::testing::IsEmpty;
using ::testing::SizeIs;

TEST(ScalabilityStructureL3T3Test, SkipT0FrameByEncoderKeepsReferencesValid) {
  std::vector<GenericFrameInfo> frames;
  ScalabilityStructureL3T3 structure;
  ScalabilityStructureWrapper wrapper(structure);

  // Only S0T0 decode target is enabled.
  structure.OnRatesUpdated(EnableTemporalLayers(/*s0=*/1, /*s1=*/0));
  // Encoder generates S0T0 key frame.
  wrapper.GenerateFrames(/*num_temporal_units=*/1, frames);
  EXPECT_THAT(frames, SizeIs(1));
  // Spatial layers 1 is enabled.
  structure.OnRatesUpdated(EnableTemporalLayers(/*s0=*/1, /*s1=*/1));
  // Encoder tries to generate S0T0 and S1T0 delta frames but they are dropped.
  structure.NextFrameConfig(/*restart=*/false);
  // Encoder successfully generates S0T0 and S1T0 delta frames.
  wrapper.GenerateFrames(/*num_temporal_units=*/1, frames);
  EXPECT_THAT(frames, SizeIs(3));

  EXPECT_TRUE(wrapper.FrameReferencesAreValid(frames));
}

TEST(ScalabilityStructureL3T3Test, SkipS1T1FrameKeepsStructureValid) {
  ScalabilityStructureL3T3 structure;
  ScalabilityStructureWrapper wrapper(structure);

  structure.OnRatesUpdated(EnableTemporalLayers(/*s0=*/3, /*s1=*/3));
  auto frames = wrapper.GenerateFrames(/*num_temporal_units=*/1);
  EXPECT_THAT(frames, SizeIs(2));
  EXPECT_EQ(frames[0].temporal_id, 0);

  frames = wrapper.GenerateFrames(/*num_temporal_units=*/1);
  EXPECT_THAT(frames, SizeIs(2));
  EXPECT_EQ(frames[0].temporal_id, 2);

  structure.OnRatesUpdated(EnableTemporalLayers(/*s0=*/3, /*s1=*/0));
  frames = wrapper.GenerateFrames(/*num_temporal_units=*/1);
  EXPECT_THAT(frames, SizeIs(1));
  EXPECT_EQ(frames[0].temporal_id, 1);

  structure.OnRatesUpdated(EnableTemporalLayers(/*s0=*/3, /*s1=*/3));
  // Rely on checks inside GenerateFrames frame references are valid.
  frames = wrapper.GenerateFrames(/*num_temporal_units=*/1);
  EXPECT_THAT(frames, SizeIs(2));
  EXPECT_EQ(frames[0].temporal_id, 2);
}

TEST(ScalabilityStructureL3T3Test, SkipT1FrameByEncoderKeepsReferencesValid) {
  std::vector<GenericFrameInfo> frames;
  ScalabilityStructureL3T3 structure;
  ScalabilityStructureWrapper wrapper(structure);

  // 1st 2 temporal units (T0 and T2)
  wrapper.GenerateFrames(/*num_temporal_units=*/2, frames);
  // Simulate T1 frame dropped by the encoder,
  // i.e. retrieve config, but skip calling OnEncodeDone.
  structure.NextFrameConfig(/*restart=*/false);
  // one more temporal units (T2)
  wrapper.GenerateFrames(/*num_temporal_units=*/1, frames);

  EXPECT_TRUE(wrapper.FrameReferencesAreValid(frames));
}

TEST(ScalabilityStructureL3T3Test,
     SkippingFrameReusePreviousFrameConfiguration) {
  std::vector<GenericFrameInfo> frames;
  ScalabilityStructureL3T3 structure;
  ScalabilityStructureWrapper wrapper(structure);

  // 1st 2 temporal units (T0 and T2)
  wrapper.GenerateFrames(/*num_temporal_units=*/2, frames);
  ASSERT_THAT(frames, SizeIs(6));
  ASSERT_EQ(frames[0].temporal_id, 0);
  ASSERT_EQ(frames[3].temporal_id, 2);

  // Simulate a frame dropped by the encoder,
  // i.e. retrieve config, but skip calling OnEncodeDone.
  structure.NextFrameConfig(/*restart=*/false);
  // two more temporal unit, expect temporal pattern continues
  wrapper.GenerateFrames(/*num_temporal_units=*/2, frames);
  ASSERT_THAT(frames, SizeIs(12));
  // Expect temporal pattern continues as if there were no dropped frames.
  EXPECT_EQ(frames[6].temporal_id, 1);
  EXPECT_EQ(frames[9].temporal_id, 2);
}

TEST(ScalabilityStructureL3T3Test, SwitchSpatialLayerBeforeT1Frame) {
  ScalabilityStructureL3T3 structure;
  ScalabilityStructureWrapper wrapper(structure);

  structure.OnRatesUpdated(EnableTemporalLayers(/*s0=*/2, /*s1=*/0));
  EXPECT_THAT(wrapper.GenerateFrames(1), SizeIs(1));
  structure.OnRatesUpdated(EnableTemporalLayers(/*s0=*/0, /*s1=*/2));
  auto frames = wrapper.GenerateFrames(1);
  ASSERT_THAT(frames, SizeIs(1));
  EXPECT_THAT(frames[0].frame_diffs, IsEmpty());
  EXPECT_EQ(frames[0].temporal_id, 0);
}

}  // namespace
}  // namespace webrtc
