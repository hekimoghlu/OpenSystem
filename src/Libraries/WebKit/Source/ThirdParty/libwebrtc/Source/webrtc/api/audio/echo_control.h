/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 31, 2024.
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
#ifndef API_AUDIO_ECHO_CONTROL_H_
#define API_AUDIO_ECHO_CONTROL_H_

#include <memory>

#include "rtc_base/checks.h"

namespace webrtc {

class AudioBuffer;

// Interface for an acoustic echo cancellation (AEC) submodule.
class EchoControl {
 public:
  // Analysis (not changing) of the render signal.
  virtual void AnalyzeRender(AudioBuffer* render) = 0;

  // Analysis (not changing) of the capture signal.
  virtual void AnalyzeCapture(AudioBuffer* capture) = 0;

  // Processes the capture signal in order to remove the echo.
  virtual void ProcessCapture(AudioBuffer* capture, bool level_change) = 0;

  // As above, but also returns the linear filter output.
  virtual void ProcessCapture(AudioBuffer* capture,
                              AudioBuffer* linear_output,
                              bool level_change) = 0;

  struct Metrics {
    double echo_return_loss;
    double echo_return_loss_enhancement;
    int delay_ms;
  };

  // Collect current metrics from the echo controller.
  virtual Metrics GetMetrics() const = 0;

  // Provides an optional external estimate of the audio buffer delay.
  virtual void SetAudioBufferDelay(int delay_ms) = 0;

  // Specifies whether the capture output will be used. The purpose of this is
  // to allow the echo controller to deactivate some of the processing when the
  // resulting output is anyway not used, for instance when the endpoint is
  // muted.
  // TODO(b/177830919): Make pure virtual.
  virtual void SetCaptureOutputUsage(bool /* capture_output_used */) {}

  // Returns wheter the signal is altered.
  virtual bool ActiveProcessing() const = 0;

  virtual ~EchoControl() {}
};

// Interface for a factory that creates EchoControllers.
class EchoControlFactory {
 public:
  virtual std::unique_ptr<EchoControl> Create(int sample_rate_hz,
                                              int num_render_channels,
                                              int num_capture_channels) = 0;

  virtual ~EchoControlFactory() = default;
};
}  // namespace webrtc

#endif  // API_AUDIO_ECHO_CONTROL_H_
