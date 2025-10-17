/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 27, 2024.
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
#include "modules/audio_processing/aec3/config_selector.h"

#include <optional>
#include <tuple>

#include "api/audio/echo_canceller3_config.h"
#include "test/gtest.h"

namespace webrtc {

class ConfigSelectorChannelsAndContentDetection
    : public ::testing::Test,
      public ::testing::WithParamInterface<std::tuple<int, bool>> {};

INSTANTIATE_TEST_SUITE_P(ConfigSelectorMultiParameters,
                         ConfigSelectorChannelsAndContentDetection,
                         ::testing::Combine(::testing::Values(1, 2, 8),
                                            ::testing::Values(false, true)));

class ConfigSelectorChannels : public ::testing::Test,
                               public ::testing::WithParamInterface<int> {};

INSTANTIATE_TEST_SUITE_P(ConfigSelectorMultiParameters,
                         ConfigSelectorChannels,
                         ::testing::Values(1, 2, 8));

TEST_P(ConfigSelectorChannelsAndContentDetection,
       MonoConfigIsSelectedWhenNoMultiChannelConfigPresent) {
  const auto [num_channels, detect_stereo_content] = GetParam();
  EchoCanceller3Config config;
  config.multi_channel.detect_stereo_content = detect_stereo_content;
  std::optional<EchoCanceller3Config> multichannel_config;

  config.delay.default_delay = config.delay.default_delay + 1;
  const size_t custom_delay_value_in_config = config.delay.default_delay;

  ConfigSelector cs(config, multichannel_config,
                    /*num_render_input_channels=*/num_channels);
  EXPECT_EQ(cs.active_config().delay.default_delay,
            custom_delay_value_in_config);

  cs.Update(/*multichannel_content=*/false);
  EXPECT_EQ(cs.active_config().delay.default_delay,
            custom_delay_value_in_config);

  cs.Update(/*multichannel_content=*/true);
  EXPECT_EQ(cs.active_config().delay.default_delay,
            custom_delay_value_in_config);
}

TEST_P(ConfigSelectorChannelsAndContentDetection,
       CorrectInitialConfigIsSelected) {
  const auto [num_channels, detect_stereo_content] = GetParam();
  EchoCanceller3Config config;
  config.multi_channel.detect_stereo_content = detect_stereo_content;
  std::optional<EchoCanceller3Config> multichannel_config = config;

  config.delay.default_delay += 1;
  const size_t custom_delay_value_in_config = config.delay.default_delay;
  multichannel_config->delay.default_delay += 2;
  const size_t custom_delay_value_in_multichannel_config =
      multichannel_config->delay.default_delay;

  ConfigSelector cs(config, multichannel_config,
                    /*num_render_input_channels=*/num_channels);

  if (num_channels == 1 || detect_stereo_content) {
    EXPECT_EQ(cs.active_config().delay.default_delay,
              custom_delay_value_in_config);
  } else {
    EXPECT_EQ(cs.active_config().delay.default_delay,
              custom_delay_value_in_multichannel_config);
  }
}

TEST_P(ConfigSelectorChannels, CorrectConfigUpdateBehavior) {
  const int num_channels = GetParam();
  EchoCanceller3Config config;
  config.multi_channel.detect_stereo_content = true;
  std::optional<EchoCanceller3Config> multichannel_config = config;

  config.delay.default_delay += 1;
  const size_t custom_delay_value_in_config = config.delay.default_delay;
  multichannel_config->delay.default_delay += 2;
  const size_t custom_delay_value_in_multichannel_config =
      multichannel_config->delay.default_delay;

  ConfigSelector cs(config, multichannel_config,
                    /*num_render_input_channels=*/num_channels);

  cs.Update(/*multichannel_content=*/false);
  EXPECT_EQ(cs.active_config().delay.default_delay,
            custom_delay_value_in_config);

  if (num_channels == 1) {
    cs.Update(/*multichannel_content=*/false);
    EXPECT_EQ(cs.active_config().delay.default_delay,
              custom_delay_value_in_config);
  } else {
    cs.Update(/*multichannel_content=*/true);
    EXPECT_EQ(cs.active_config().delay.default_delay,
              custom_delay_value_in_multichannel_config);
  }
}

}  // namespace webrtc
