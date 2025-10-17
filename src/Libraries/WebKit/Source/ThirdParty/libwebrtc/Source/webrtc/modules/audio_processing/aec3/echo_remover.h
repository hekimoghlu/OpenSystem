/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_ECHO_REMOVER_H_
#define MODULES_AUDIO_PROCESSING_AEC3_ECHO_REMOVER_H_

#include <optional>
#include <vector>

#include "api/audio/echo_canceller3_config.h"
#include "api/audio/echo_control.h"
#include "modules/audio_processing/aec3/block.h"
#include "modules/audio_processing/aec3/delay_estimate.h"
#include "modules/audio_processing/aec3/echo_path_variability.h"
#include "modules/audio_processing/aec3/render_buffer.h"

namespace webrtc {

// Class for removing the echo from the capture signal.
class EchoRemover {
 public:
  static EchoRemover* Create(const EchoCanceller3Config& config,
                             int sample_rate_hz,
                             size_t num_render_channels,
                             size_t num_capture_channels);
  virtual ~EchoRemover() = default;

  // Get current metrics.
  virtual void GetMetrics(EchoControl::Metrics* metrics) const = 0;

  // Removes the echo from a block of samples from the capture signal. The
  // supplied render signal is assumed to be pre-aligned with the capture
  // signal.
  virtual void ProcessCapture(
      EchoPathVariability echo_path_variability,
      bool capture_signal_saturation,
      const std::optional<DelayEstimate>& external_delay,
      RenderBuffer* render_buffer,
      Block* linear_output,
      Block* capture) = 0;

  // Updates the status on whether echo leakage is detected in the output of the
  // echo remover.
  virtual void UpdateEchoLeakageStatus(bool leakage_detected) = 0;

  // Specifies whether the capture output will be used. The purpose of this is
  // to allow the echo remover to deactivate some of the processing when the
  // resulting output is anyway not used, for instance when the endpoint is
  // muted.
  virtual void SetCaptureOutputUsage(bool capture_output_used) = 0;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_ECHO_REMOVER_H_
