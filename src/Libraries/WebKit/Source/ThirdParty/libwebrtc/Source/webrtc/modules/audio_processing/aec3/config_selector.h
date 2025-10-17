/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_CONFIG_SELECTOR_H_
#define MODULES_AUDIO_PROCESSING_AEC3_CONFIG_SELECTOR_H_

#include <optional>

#include "api/audio/echo_canceller3_config.h"

namespace webrtc {

// Selects the config to use.
class ConfigSelector {
 public:
  ConfigSelector(const EchoCanceller3Config& config,
                 const std::optional<EchoCanceller3Config>& multichannel_config,
                 int num_render_input_channels);

  // Updates the config selection based on the detection of multichannel
  // content.
  void Update(bool multichannel_content);

  const EchoCanceller3Config& active_config() const { return *active_config_; }

 private:
  const EchoCanceller3Config config_;
  const std::optional<EchoCanceller3Config> multichannel_config_;
  const EchoCanceller3Config* active_config_ = nullptr;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_CONFIG_SELECTOR_H_
