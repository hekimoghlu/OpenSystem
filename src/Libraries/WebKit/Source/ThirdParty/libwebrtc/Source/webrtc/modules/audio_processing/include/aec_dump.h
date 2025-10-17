/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_INCLUDE_AEC_DUMP_H_
#define MODULES_AUDIO_PROCESSING_INCLUDE_AEC_DUMP_H_

#include <stdint.h>

#include <optional>
#include <string>

#include "absl/base/attributes.h"
#include "api/audio/audio_processing.h"
#include "modules/audio_processing/include/audio_frame_view.h"

namespace webrtc {

// Struct for passing current config from APM without having to
// include protobuf headers.
struct InternalAPMConfig {
  InternalAPMConfig();
  InternalAPMConfig(const InternalAPMConfig&);
  InternalAPMConfig(InternalAPMConfig&&);

  InternalAPMConfig& operator=(const InternalAPMConfig&);
  InternalAPMConfig& operator=(InternalAPMConfig&&) = delete;

  bool operator==(const InternalAPMConfig& other) const;

  bool aec_enabled = false;
  bool aec_delay_agnostic_enabled = false;
  bool aec_drift_compensation_enabled = false;
  bool aec_extended_filter_enabled = false;
  int aec_suppression_level = 0;
  bool aecm_enabled = false;
  bool aecm_comfort_noise_enabled = false;
  int aecm_routing_mode = 0;
  bool agc_enabled = false;
  int agc_mode = 0;
  bool agc_limiter_enabled = false;
  bool hpf_enabled = false;
  bool ns_enabled = false;
  int ns_level = 0;
  bool transient_suppression_enabled = false;
  bool noise_robust_agc_enabled = false;
  bool pre_amplifier_enabled = false;
  float pre_amplifier_fixed_gain_factor = 1.f;
  std::string experiments_description = "";
};

// An interface for recording configuration and input/output streams
// of the Audio Processing Module. The recordings are called
// 'aec-dumps' and are stored in a protobuf format defined in
// debug.proto.
// The Write* methods are always safe to call concurrently or
// otherwise for all implementing subclasses. The intended mode of
// operation is to create a protobuf object from the input, and send
// it away to be written to file asynchronously.
class AecDump {
 public:
  struct AudioProcessingState {
    int delay;
    int drift;
    std::optional<int> applied_input_volume;
    bool keypress;
  };

  virtual ~AecDump() = default;

  // Logs Event::Type INIT message.
  virtual void WriteInitMessage(const ProcessingConfig& api_format,
                                int64_t time_now_ms) = 0;
  ABSL_DEPRECATED("")
  void WriteInitMessage(const ProcessingConfig& api_format) {
    WriteInitMessage(api_format, 0);
  }

  // Logs Event::Type STREAM message. To log an input/output pair,
  // call the AddCapture* and AddAudioProcessingState methods followed
  // by a WriteCaptureStreamMessage call.
  virtual void AddCaptureStreamInput(
      const AudioFrameView<const float>& src) = 0;
  virtual void AddCaptureStreamOutput(
      const AudioFrameView<const float>& src) = 0;
  virtual void AddCaptureStreamInput(const int16_t* const data,
                                     int num_channels,
                                     int samples_per_channel) = 0;
  virtual void AddCaptureStreamOutput(const int16_t* const data,
                                      int num_channels,
                                      int samples_per_channel) = 0;
  virtual void AddAudioProcessingState(const AudioProcessingState& state) = 0;
  virtual void WriteCaptureStreamMessage() = 0;

  // Logs Event::Type REVERSE_STREAM message.
  virtual void WriteRenderStreamMessage(const int16_t* const data,
                                        int num_channels,
                                        int samples_per_channel) = 0;
  virtual void WriteRenderStreamMessage(
      const AudioFrameView<const float>& src) = 0;

  virtual void WriteRuntimeSetting(
      const AudioProcessing::RuntimeSetting& runtime_setting) = 0;

  // Logs Event::Type CONFIG message.
  virtual void WriteConfig(const InternalAPMConfig& config) = 0;
};
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_INCLUDE_AEC_DUMP_H_
