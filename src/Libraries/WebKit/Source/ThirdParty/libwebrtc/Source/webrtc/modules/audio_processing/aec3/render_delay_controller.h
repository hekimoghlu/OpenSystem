/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 17, 2021.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_RENDER_DELAY_CONTROLLER_H_
#define MODULES_AUDIO_PROCESSING_AEC3_RENDER_DELAY_CONTROLLER_H_

#include <optional>

#include "api/array_view.h"
#include "api/audio/echo_canceller3_config.h"
#include "modules/audio_processing/aec3/block.h"
#include "modules/audio_processing/aec3/delay_estimate.h"
#include "modules/audio_processing/aec3/downsampled_render_buffer.h"
#include "modules/audio_processing/aec3/render_delay_buffer.h"
#include "modules/audio_processing/logging/apm_data_dumper.h"

namespace webrtc {

// Class for aligning the render and capture signal using a RenderDelayBuffer.
class RenderDelayController {
 public:
  static RenderDelayController* Create(const EchoCanceller3Config& config,
                                       int sample_rate_hz,
                                       size_t num_capture_channels);
  virtual ~RenderDelayController() = default;

  // Resets the delay controller. If the delay confidence is reset, the reset
  // behavior is as if the call is restarted.
  virtual void Reset(bool reset_delay_confidence) = 0;

  // Logs a render call.
  virtual void LogRenderCall() = 0;

  // Aligns the render buffer content with the capture signal.
  virtual std::optional<DelayEstimate> GetDelay(
      const DownsampledRenderBuffer& render_buffer,
      size_t render_delay_buffer_delay,
      const Block& capture) = 0;

  // Returns true if clockdrift has been detected.
  virtual bool HasClockdrift() const = 0;
};
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_RENDER_DELAY_CONTROLLER_H_
