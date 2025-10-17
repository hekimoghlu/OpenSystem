/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 16, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_ECHO_CONTROL_MOBILE_IMPL_H_
#define MODULES_AUDIO_PROCESSING_ECHO_CONTROL_MOBILE_IMPL_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <vector>

#include "api/array_view.h"

namespace webrtc {

class AudioBuffer;

// The acoustic echo control for mobile (AECM) component is a low complexity
// robust option intended for use on mobile devices.
class EchoControlMobileImpl {
 public:
  EchoControlMobileImpl();

  ~EchoControlMobileImpl();

  // Recommended settings for particular audio routes. In general, the louder
  // the echo is expected to be, the higher this value should be set. The
  // preferred setting may vary from device to device.
  enum RoutingMode {
    kQuietEarpieceOrHeadset,
    kEarpiece,
    kLoudEarpiece,
    kSpeakerphone,
    kLoudSpeakerphone
  };

  // Sets echo control appropriate for the audio routing `mode` on the device.
  // It can and should be updated during a call if the audio routing changes.
  int set_routing_mode(RoutingMode mode);
  RoutingMode routing_mode() const;

  // Comfort noise replaces suppressed background noise to maintain a
  // consistent signal level.
  int enable_comfort_noise(bool enable);
  bool is_comfort_noise_enabled() const;

  void ProcessRenderAudio(rtc::ArrayView<const int16_t> packed_render_audio);
  int ProcessCaptureAudio(AudioBuffer* audio, int stream_delay_ms);

  void Initialize(int sample_rate_hz,
                  size_t num_reverse_channels,
                  size_t num_output_channels);

  static void PackRenderAudioBuffer(const AudioBuffer* audio,
                                    size_t num_output_channels,
                                    size_t num_channels,
                                    std::vector<int16_t>* packed_buffer);

  static size_t NumCancellersRequired(size_t num_output_channels,
                                      size_t num_reverse_channels);

 private:
  class Canceller;
  struct StreamProperties;

  int Configure();

  RoutingMode routing_mode_;
  bool comfort_noise_enabled_;

  std::vector<std::unique_ptr<Canceller>> cancellers_;
  std::unique_ptr<StreamProperties> stream_properties_;
  std::vector<std::array<int16_t, 160>> low_pass_reference_;
  bool reference_copied_ = false;
};
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_ECHO_CONTROL_MOBILE_IMPL_H_
