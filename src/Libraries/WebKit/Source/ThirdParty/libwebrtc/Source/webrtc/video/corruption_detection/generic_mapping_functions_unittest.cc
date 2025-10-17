/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 29, 2022.
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
#include "video/corruption_detection/generic_mapping_functions.h"

#include "api/video/video_codec_type.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

using ::testing::DoubleNear;
using ::testing::FieldsAre;

constexpr double kMaxAbsoluteError = 1e-4;

constexpr int kLumaThreshold = 5;
constexpr int kChromaThresholdVp8 = 6;
constexpr int kChromaThresholdVp9 = 4;
constexpr int kChromaThresholdAv1 = 4;
constexpr int kChromaThresholdH264 = 2;

TEST(GenericMappingFunctionsTest, TestVp8) {
  constexpr VideoCodecType kCodecType = VideoCodecType::kVideoCodecVP8;
  EXPECT_THAT(GetCorruptionFilterSettings(
                  /*qp=*/10, kCodecType),
              FieldsAre(DoubleNear(0.5139, kMaxAbsoluteError), kLumaThreshold,
                        kChromaThresholdVp8));
  EXPECT_THAT(GetCorruptionFilterSettings(
                  /*qp=*/100, kCodecType),
              FieldsAre(DoubleNear(2.7351, kMaxAbsoluteError), kLumaThreshold,
                        kChromaThresholdVp8));
  EXPECT_THAT(GetCorruptionFilterSettings(/*qp=*/127, kCodecType),
              FieldsAre(DoubleNear(4.5162, kMaxAbsoluteError), kLumaThreshold,
                        kChromaThresholdVp8));
}

TEST(GenericMappingFunctionsTest, TestVp9) {
  constexpr VideoCodecType kCodecType = VideoCodecType::kVideoCodecVP9;
  EXPECT_THAT(GetCorruptionFilterSettings(/*qp=*/10, kCodecType),
              FieldsAre(DoubleNear(0.3405, kMaxAbsoluteError), kLumaThreshold,
                        kChromaThresholdVp9));
  EXPECT_THAT(GetCorruptionFilterSettings(/*qp=*/100, kCodecType),
              FieldsAre(DoubleNear(0.9369, kMaxAbsoluteError), kLumaThreshold,
                        kChromaThresholdVp9));
  EXPECT_THAT(GetCorruptionFilterSettings(/*qp=*/200, kCodecType),
              FieldsAre(DoubleNear(3.8088, kMaxAbsoluteError), kLumaThreshold,
                        kChromaThresholdVp9));
  EXPECT_THAT(GetCorruptionFilterSettings(/*qp=*/255, kCodecType),
              FieldsAre(DoubleNear(127.8, kMaxAbsoluteError), kLumaThreshold,
                        kChromaThresholdVp9));
}

TEST(GenericMappingFunctionsTest, TestAv1) {
  constexpr VideoCodecType kCodecType = VideoCodecType::kVideoCodecAV1;
  EXPECT_THAT(GetCorruptionFilterSettings(/*qp=*/10, kCodecType),
              FieldsAre(DoubleNear(0.4480, kMaxAbsoluteError), kLumaThreshold,
                        kChromaThresholdAv1));
  EXPECT_THAT(GetCorruptionFilterSettings(/*qp=*/100, kCodecType),
              FieldsAre(DoubleNear(0.8623, kMaxAbsoluteError), kLumaThreshold,
                        kChromaThresholdAv1));
  EXPECT_THAT(GetCorruptionFilterSettings(/*qp=*/200, kCodecType),
              FieldsAre(DoubleNear(2.8842, kMaxAbsoluteError), kLumaThreshold,
                        kChromaThresholdAv1));
  EXPECT_THAT(GetCorruptionFilterSettings(/*qp=*/255, kCodecType),
              FieldsAre(DoubleNear(176.37, kMaxAbsoluteError), kLumaThreshold,
                        kChromaThresholdAv1));
}

TEST(GenericMappingFunctionsTest, TestH264) {
  constexpr VideoCodecType kCodecType = VideoCodecType::kVideoCodecH264;
  EXPECT_THAT(GetCorruptionFilterSettings(/*qp=*/10, kCodecType),
              FieldsAre(DoubleNear(0.263, kMaxAbsoluteError), kLumaThreshold,
                        kChromaThresholdH264));
  EXPECT_THAT(GetCorruptionFilterSettings(/*qp=*/30, kCodecType),
              FieldsAre(DoubleNear(4.3047, kMaxAbsoluteError), kLumaThreshold,
                        kChromaThresholdH264));
  EXPECT_THAT(GetCorruptionFilterSettings(/*qp=*/51, kCodecType),
              FieldsAre(DoubleNear(81.0346, kMaxAbsoluteError), kLumaThreshold,
                        kChromaThresholdH264));
}

}  // namespace
}  // namespace webrtc
