/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 13, 2023.
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
#include "api/audio/echo_canceller3_config.h"

#include "modules/audio_processing/test/echo_canceller3_config_json.h"
#include "test/gtest.h"

namespace webrtc {

TEST(EchoCanceller3Config, ValidConfigIsNotModified) {
  EchoCanceller3Config config;
  EXPECT_TRUE(EchoCanceller3Config::Validate(&config));
  EchoCanceller3Config default_config;
  EXPECT_EQ(Aec3ConfigToJsonString(config),
            Aec3ConfigToJsonString(default_config));
}

TEST(EchoCanceller3Config, InvalidConfigIsCorrected) {
  // Change a parameter and validate.
  EchoCanceller3Config config;
  config.echo_model.min_noise_floor_power = -1600000.f;
  EXPECT_FALSE(EchoCanceller3Config::Validate(&config));
  EXPECT_GE(config.echo_model.min_noise_floor_power, 0.f);
  // Verify remaining parameters are unchanged.
  EchoCanceller3Config default_config;
  config.echo_model.min_noise_floor_power =
      default_config.echo_model.min_noise_floor_power;
  EXPECT_EQ(Aec3ConfigToJsonString(config),
            Aec3ConfigToJsonString(default_config));
}

TEST(EchoCanceller3Config, ValidatedConfigsAreValid) {
  EchoCanceller3Config config;
  config.delay.down_sampling_factor = 983;
  EXPECT_FALSE(EchoCanceller3Config::Validate(&config));
  EXPECT_TRUE(EchoCanceller3Config::Validate(&config));
}
}  // namespace webrtc
