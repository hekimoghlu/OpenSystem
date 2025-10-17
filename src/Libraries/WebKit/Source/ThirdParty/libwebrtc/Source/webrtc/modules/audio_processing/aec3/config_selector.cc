/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 20, 2022.
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

#include "rtc_base/checks.h"

namespace webrtc {
namespace {

// Validates that the mono and the multichannel configs have compatible fields.
bool CompatibleConfigs(const EchoCanceller3Config& mono_config,
                       const EchoCanceller3Config& multichannel_config) {
  if (mono_config.delay.fixed_capture_delay_samples !=
      multichannel_config.delay.fixed_capture_delay_samples) {
    return false;
  }
  if (mono_config.filter.export_linear_aec_output !=
      multichannel_config.filter.export_linear_aec_output) {
    return false;
  }
  if (mono_config.filter.high_pass_filter_echo_reference !=
      multichannel_config.filter.high_pass_filter_echo_reference) {
    return false;
  }
  if (mono_config.multi_channel.detect_stereo_content !=
      multichannel_config.multi_channel.detect_stereo_content) {
    return false;
  }
  if (mono_config.multi_channel.stereo_detection_timeout_threshold_seconds !=
      multichannel_config.multi_channel
          .stereo_detection_timeout_threshold_seconds) {
    return false;
  }
  return true;
}

}  // namespace

ConfigSelector::ConfigSelector(
    const EchoCanceller3Config& config,
    const std::optional<EchoCanceller3Config>& multichannel_config,
    int num_render_input_channels)
    : config_(config), multichannel_config_(multichannel_config) {
  if (multichannel_config_.has_value()) {
    RTC_DCHECK(CompatibleConfigs(config_, *multichannel_config_));
  }

  Update(!config_.multi_channel.detect_stereo_content &&
         num_render_input_channels > 1);

  RTC_DCHECK(active_config_);
}

void ConfigSelector::Update(bool multichannel_content) {
  if (multichannel_content && multichannel_config_.has_value()) {
    active_config_ = &(*multichannel_config_);
  } else {
    active_config_ = &config_;
  }
}

}  // namespace webrtc
