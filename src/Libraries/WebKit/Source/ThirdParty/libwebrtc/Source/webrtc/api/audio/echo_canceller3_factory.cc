/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 21, 2024.
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
#include "api/audio/echo_canceller3_factory.h"

#include <memory>
#include <optional>

#include "api/audio/echo_canceller3_config.h"
#include "api/audio/echo_control.h"
#include "modules/audio_processing/aec3/echo_canceller3.h"

namespace webrtc {

EchoCanceller3Factory::EchoCanceller3Factory() {}

EchoCanceller3Factory::EchoCanceller3Factory(const EchoCanceller3Config& config)
    : config_(config) {}

std::unique_ptr<EchoControl> EchoCanceller3Factory::Create(
    int sample_rate_hz,
    int num_render_channels,
    int num_capture_channels) {
  return std::make_unique<EchoCanceller3>(
      config_, /*multichannel_config=*/std::nullopt, sample_rate_hz,
      num_render_channels, num_capture_channels);
}

}  // namespace webrtc
