/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 28, 2022.
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
#include "rtc_base/experiments/normalize_simulcast_size_experiment.h"

#include "test/explicit_key_value_config.h"
#include "test/gtest.h"

namespace webrtc {

using test::ExplicitKeyValueConfig;

TEST(NormalizeSimulcastSizeExperimentTest, GetExponent) {
  ExplicitKeyValueConfig field_trials(
      "WebRTC-NormalizeSimulcastResolution/Enabled-2/");
  EXPECT_EQ(2,
            NormalizeSimulcastSizeExperiment::GetBase2Exponent(field_trials));
}

TEST(NormalizeSimulcastSizeExperimentTest, GetExponentWithTwoParameters) {
  ExplicitKeyValueConfig field_trials(
      "WebRTC-NormalizeSimulcastResolution/Enabled-3-4/");
  EXPECT_EQ(3,
            NormalizeSimulcastSizeExperiment::GetBase2Exponent(field_trials));
}

TEST(NormalizeSimulcastSizeExperimentTest, GetExponentFailsIfNotEnabled) {
  ExplicitKeyValueConfig field_trials(
      "WebRTC-NormalizeSimulcastResolution/Disabled/");
  EXPECT_FALSE(
      NormalizeSimulcastSizeExperiment::GetBase2Exponent(field_trials));
}

TEST(NormalizeSimulcastSizeExperimentTest,
     GetExponentFailsForInvalidFieldTrial) {
  ExplicitKeyValueConfig field_trials(
      "WebRTC-NormalizeSimulcastResolution/Enabled-invalid/");
  EXPECT_FALSE(
      NormalizeSimulcastSizeExperiment::GetBase2Exponent(field_trials));
}

TEST(NormalizeSimulcastSizeExperimentTest,
     GetExponentFailsForNegativeOutOfBoundValue) {
  // Supported range: [0, 5].
  ExplicitKeyValueConfig field_trials(
      "WebRTC-NormalizeSimulcastResolution/Enabled--1/");
  EXPECT_FALSE(
      NormalizeSimulcastSizeExperiment::GetBase2Exponent(field_trials));
}

TEST(NormalizeSimulcastSizeExperimentTest,
     GetExponentFailsForPositiveOutOfBoundValue) {
  // Supported range: [0, 5].
  ExplicitKeyValueConfig field_trials(
      "WebRTC-NormalizeSimulcastResolution/Enabled-6/");
  EXPECT_FALSE(
      NormalizeSimulcastSizeExperiment::GetBase2Exponent(field_trials));
}

}  // namespace webrtc
