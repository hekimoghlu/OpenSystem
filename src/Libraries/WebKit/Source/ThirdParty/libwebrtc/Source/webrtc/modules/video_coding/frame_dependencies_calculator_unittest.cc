/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
#include "modules/video_coding/frame_dependencies_calculator.h"

#include "common_video/generic_frame_descriptor/generic_frame_info.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

constexpr CodecBufferUsage ReferenceAndUpdate(int id) {
  return CodecBufferUsage(id, /*referenced=*/true, /*updated=*/true);
}
constexpr CodecBufferUsage Reference(int id) {
  return CodecBufferUsage(id, /*referenced=*/true, /*updated=*/false);
}
constexpr CodecBufferUsage Update(int id) {
  return CodecBufferUsage(id, /*referenced=*/false, /*updated=*/true);
}

TEST(FrameDependenciesCalculatorTest, SingleLayer) {
  CodecBufferUsage pattern[] = {ReferenceAndUpdate(0)};
  FrameDependenciesCalculator calculator;

  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/1, pattern), IsEmpty());
  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/3, pattern),
              ElementsAre(1));
  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/6, pattern),
              ElementsAre(3));
}

TEST(FrameDependenciesCalculatorTest, TwoTemporalLayers) {
  // Shortened 4-frame pattern:
  // T1:  2---4   6---8 ...
  //      /   /   /   /
  // T0: 1---3---5---7 ...
  CodecBufferUsage pattern0[] = {ReferenceAndUpdate(0)};
  CodecBufferUsage pattern1[] = {Reference(0), Update(1)};
  CodecBufferUsage pattern2[] = {ReferenceAndUpdate(0)};
  CodecBufferUsage pattern3[] = {Reference(0), Reference(1)};
  FrameDependenciesCalculator calculator;

  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/1, pattern0), IsEmpty());
  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/2, pattern1),
              ElementsAre(1));
  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/3, pattern2),
              ElementsAre(1));
  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/4, pattern3),
              UnorderedElementsAre(2, 3));
  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/5, pattern0),
              ElementsAre(3));
  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/6, pattern1),
              ElementsAre(5));
  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/7, pattern2),
              ElementsAre(5));
  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/8, pattern3),
              UnorderedElementsAre(6, 7));
}

TEST(FrameDependenciesCalculatorTest, ThreeTemporalLayers4FramePattern) {
  // T2:   2---4   6---8 ...
  //      /   /   /   /
  // T1:  |  3    |  7   ...
  //      /_/     /_/
  // T0: 1-------5-----  ...
  CodecBufferUsage pattern0[] = {ReferenceAndUpdate(0)};
  CodecBufferUsage pattern1[] = {Reference(0), Update(2)};
  CodecBufferUsage pattern2[] = {Reference(0), Update(1)};
  CodecBufferUsage pattern3[] = {Reference(0), Reference(1), Reference(2)};
  FrameDependenciesCalculator calculator;

  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/1, pattern0), IsEmpty());
  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/2, pattern1),
              ElementsAre(1));
  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/3, pattern2),
              ElementsAre(1));
  // Note that frame#4 references buffer#0 that is updated by frame#1,
  // yet there is no direct dependency from frame#4 to frame#1.
  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/4, pattern3),
              UnorderedElementsAre(2, 3));
  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/5, pattern0),
              ElementsAre(1));
  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/6, pattern1),
              ElementsAre(5));
}

TEST(FrameDependenciesCalculatorTest, SimulcastWith2Layers) {
  // S1: 2---4---6-  ...
  //
  // S0: 1---3---5-  ...
  CodecBufferUsage pattern0[] = {ReferenceAndUpdate(0)};
  CodecBufferUsage pattern1[] = {ReferenceAndUpdate(1)};
  FrameDependenciesCalculator calculator;

  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/1, pattern0), IsEmpty());
  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/2, pattern1), IsEmpty());
  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/3, pattern0),
              ElementsAre(1));
  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/4, pattern1),
              ElementsAre(2));
  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/5, pattern0),
              ElementsAre(3));
  EXPECT_THAT(calculator.FromBuffersUsage(/*frame_id=*/6, pattern1),
              ElementsAre(4));
}

}  // namespace
}  // namespace webrtc
