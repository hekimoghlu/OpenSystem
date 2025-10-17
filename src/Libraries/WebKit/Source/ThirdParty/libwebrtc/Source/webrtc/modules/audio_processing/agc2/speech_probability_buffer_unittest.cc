/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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
#include "modules/audio_processing/agc2/speech_probability_buffer.h"

#include <algorithm>

#include "test/gtest.h"

namespace webrtc {
namespace {

constexpr float kAbsError = 0.001f;
constexpr float kActivityThreshold = 0.9f;
constexpr float kLowProbabilityThreshold = 0.2f;
constexpr int kNumAnalysisFrames = 100;

}  // namespace

TEST(SpeechProbabilityBufferTest, CheckSumAfterInitialization) {
  SpeechProbabilityBuffer buffer(kLowProbabilityThreshold);

  EXPECT_EQ(buffer.GetSumProbabilities(), 0.0f);
}

TEST(SpeechProbabilityBufferTest, CheckSumAfterUpdate) {
  SpeechProbabilityBuffer buffer(kLowProbabilityThreshold);

  buffer.Update(0.7f);

  EXPECT_NEAR(buffer.GetSumProbabilities(), 0.7f, kAbsError);

  buffer.Update(0.6f);

  EXPECT_NEAR(buffer.GetSumProbabilities(), 1.3f, kAbsError);

  for (int i = 0; i < kNumAnalysisFrames - 1; ++i) {
    buffer.Update(1.0f);
  }

  EXPECT_NEAR(buffer.GetSumProbabilities(), 99.6f, kAbsError);
}

TEST(SpeechProbabilityBufferTest, CheckSumAfterReset) {
  SpeechProbabilityBuffer buffer(kLowProbabilityThreshold);

  buffer.Update(0.7f);
  buffer.Update(0.6f);
  buffer.Update(0.3f);

  EXPECT_GT(buffer.GetSumProbabilities(), 0.0f);

  buffer.Reset();

  EXPECT_EQ(buffer.GetSumProbabilities(), 0.0f);
}

TEST(SpeechProbabilityBufferTest, CheckSumAfterTransientNotRemoved) {
  SpeechProbabilityBuffer buffer(kLowProbabilityThreshold);

  buffer.Update(1.0f);
  buffer.Update(1.0f);
  buffer.Update(1.0f);
  buffer.Update(1.0f);
  buffer.Update(1.0f);
  buffer.Update(1.0f);
  buffer.Update(1.0f);
  buffer.Update(1.0f);
  buffer.Update(1.0f);

  buffer.Update(0.0f);

  EXPECT_NEAR(buffer.GetSumProbabilities(), 9.0f, kAbsError);

  buffer.Update(0.0f);

  EXPECT_NEAR(buffer.GetSumProbabilities(), 9.0f, kAbsError);
}

TEST(SpeechProbabilityBufferTest, CheckSumAfterTransientRemoved) {
  SpeechProbabilityBuffer buffer(kLowProbabilityThreshold);

  buffer.Update(0.0f);
  buffer.Update(0.0f);
  buffer.Update(0.0f);
  buffer.Update(0.0f);
  buffer.Update(0.0f);
  buffer.Update(0.0f);
  buffer.Update(1.0f);
  buffer.Update(1.0f);
  buffer.Update(1.0f);

  EXPECT_NEAR(buffer.GetSumProbabilities(), 3.0f, kAbsError);

  buffer.Update(0.0f);

  EXPECT_NEAR(buffer.GetSumProbabilities(), 0.0f, kAbsError);
}

TEST(SpeechProbabilityBufferTest, CheckSegmentIsNotActiveAfterNoUpdates) {
  SpeechProbabilityBuffer buffer(kLowProbabilityThreshold);

  EXPECT_FALSE(buffer.IsActiveSegment());
}

TEST(SpeechProbabilityBufferTest, CheckSegmentIsActiveChangesFromFalseToTrue) {
  SpeechProbabilityBuffer buffer(kLowProbabilityThreshold);

  // Add low probabilities until the buffer is full. That's not enough
  // to make `IsActiveSegment()` to return true.
  for (int i = 0; i < kNumAnalysisFrames; ++i) {
    buffer.Update(0.0f);
  }

  EXPECT_FALSE(buffer.IsActiveSegment());

  // Add high probabilities until `IsActiveSegment()` returns true.
  for (int i = 0; i < kActivityThreshold * kNumAnalysisFrames - 1; ++i) {
    buffer.Update(1.0f);
  }

  EXPECT_FALSE(buffer.IsActiveSegment());

  buffer.Update(1.0f);

  EXPECT_TRUE(buffer.IsActiveSegment());
}

TEST(SpeechProbabilityBufferTest, CheckSegmentIsActiveChangesFromTrueToFalse) {
  SpeechProbabilityBuffer buffer(kLowProbabilityThreshold);

  // Add high probabilities until the buffer is full. That's enough to
  // make `IsActiveSegment()` to return true.
  for (int i = 0; i < kNumAnalysisFrames; ++i) {
    buffer.Update(1.0f);
  }

  EXPECT_TRUE(buffer.IsActiveSegment());

  // Add low probabilities until `IsActiveSegment()` returns false.
  for (int i = 0; i < (1.0f - kActivityThreshold) * kNumAnalysisFrames - 1;
       ++i) {
    buffer.Update(0.0f);
  }

  EXPECT_TRUE(buffer.IsActiveSegment());

  buffer.Update(1.0f);

  EXPECT_TRUE(buffer.IsActiveSegment());
}

TEST(SpeechProbabilityBufferTest,
     CheckSegmentIsActiveAfterUpdatesWithHighProbabilities) {
  SpeechProbabilityBuffer buffer(kLowProbabilityThreshold);

  for (int i = 0; i < kNumAnalysisFrames - 1; ++i) {
    buffer.Update(1.0f);
  }

  EXPECT_FALSE(buffer.IsActiveSegment());

  buffer.Update(1.0f);

  EXPECT_TRUE(buffer.IsActiveSegment());
}

TEST(SpeechProbabilityBufferTest,
     CheckSegmentIsNotActiveAfterUpdatesWithLowProbabilities) {
  SpeechProbabilityBuffer buffer(kLowProbabilityThreshold);

  for (int i = 0; i < kNumAnalysisFrames - 1; ++i) {
    buffer.Update(0.3f);
  }

  EXPECT_FALSE(buffer.IsActiveSegment());

  buffer.Update(0.3f);

  EXPECT_FALSE(buffer.IsActiveSegment());
}

TEST(SpeechProbabilityBufferTest, CheckSegmentIsActiveAfterBufferIsFull) {
  SpeechProbabilityBuffer buffer(kLowProbabilityThreshold);

  for (int i = 0; i < kNumAnalysisFrames - 1; ++i) {
    buffer.Update(1.0f);
  }

  EXPECT_FALSE(buffer.IsActiveSegment());

  buffer.Update(1.0f);

  EXPECT_TRUE(buffer.IsActiveSegment());

  buffer.Update(1.0f);

  EXPECT_TRUE(buffer.IsActiveSegment());
}

TEST(SpeechProbabilityBufferTest, CheckSegmentIsNotActiveAfterBufferIsFull) {
  SpeechProbabilityBuffer buffer(kLowProbabilityThreshold);

  for (int i = 0; i < kNumAnalysisFrames - 1; ++i) {
    buffer.Update(0.29f);
  }

  EXPECT_FALSE(buffer.IsActiveSegment());

  buffer.Update(0.29f);

  EXPECT_FALSE(buffer.IsActiveSegment());

  buffer.Update(0.29f);

  EXPECT_FALSE(buffer.IsActiveSegment());
}

TEST(SpeechProbabilityBufferTest, CheckSegmentIsNotActiveAfterReset) {
  SpeechProbabilityBuffer buffer(kLowProbabilityThreshold);

  for (int i = 0; i < kNumAnalysisFrames; ++i) {
    buffer.Update(1.0f);
  }

  EXPECT_TRUE(buffer.IsActiveSegment());

  buffer.Reset();

  EXPECT_FALSE(buffer.IsActiveSegment());
}

TEST(SpeechProbabilityBufferTest,
     CheckSegmentIsNotActiveAfterTransientRemovedAfterFewUpdates) {
  SpeechProbabilityBuffer buffer(kLowProbabilityThreshold);

  buffer.Update(0.4f);
  buffer.Update(0.4f);
  buffer.Update(0.0f);

  EXPECT_FALSE(buffer.IsActiveSegment());
}

TEST(SpeechProbabilityBufferTest,
     CheckSegmentIsActiveAfterTransientNotRemoved) {
  SpeechProbabilityBuffer buffer(kLowProbabilityThreshold);

  for (int i = 0; i < kNumAnalysisFrames; ++i) {
    buffer.Update(1.0f);
  }

  buffer.Update(0.7f);
  buffer.Update(0.8f);
  buffer.Update(0.9f);
  buffer.Update(1.0f);

  EXPECT_TRUE(buffer.IsActiveSegment());

  buffer.Update(0.0f);

  EXPECT_TRUE(buffer.IsActiveSegment());

  buffer.Update(0.7f);

  EXPECT_TRUE(buffer.IsActiveSegment());
}

TEST(SpeechProbabilityBufferTest,
     CheckSegmentIsNotActiveAfterTransientNotRemoved) {
  SpeechProbabilityBuffer buffer(kLowProbabilityThreshold);

  for (int i = 0; i < kNumAnalysisFrames; ++i) {
    buffer.Update(0.1f);
  }

  buffer.Update(0.7f);
  buffer.Update(0.8f);
  buffer.Update(0.9f);
  buffer.Update(1.0f);

  EXPECT_FALSE(buffer.IsActiveSegment());

  buffer.Update(0.0f);

  EXPECT_FALSE(buffer.IsActiveSegment());

  buffer.Update(0.7f);

  EXPECT_FALSE(buffer.IsActiveSegment());
}

TEST(SpeechProbabilityBufferTest,
     CheckSegmentIsNotActiveAfterTransientRemoved) {
  SpeechProbabilityBuffer buffer(kLowProbabilityThreshold);

  for (int i = 0; i < kNumAnalysisFrames; ++i) {
    buffer.Update(0.1f);
  }

  buffer.Update(0.7f);
  buffer.Update(0.8f);
  buffer.Update(0.9f);
  buffer.Update(1.0f);

  EXPECT_FALSE(buffer.IsActiveSegment());

  buffer.Update(0.0f);

  EXPECT_FALSE(buffer.IsActiveSegment());

  buffer.Update(0.7f);

  EXPECT_FALSE(buffer.IsActiveSegment());
}

TEST(SpeechProbabilityBufferTest, CheckSegmentIsActiveAfterTransientRemoved) {
  SpeechProbabilityBuffer buffer(kLowProbabilityThreshold);

  for (int i = 0; i < kNumAnalysisFrames; ++i) {
    buffer.Update(1.0f);
  }

  buffer.Update(0.7f);
  buffer.Update(0.8f);
  buffer.Update(0.9f);
  buffer.Update(1.0f);

  EXPECT_TRUE(buffer.IsActiveSegment());

  buffer.Update(0.0f);

  EXPECT_TRUE(buffer.IsActiveSegment());

  buffer.Update(0.7f);

  EXPECT_TRUE(buffer.IsActiveSegment());
}

}  // namespace webrtc
