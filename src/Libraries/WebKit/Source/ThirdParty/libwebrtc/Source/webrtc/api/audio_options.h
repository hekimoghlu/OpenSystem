/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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
#ifndef API_AUDIO_OPTIONS_H_
#define API_AUDIO_OPTIONS_H_

#include <optional>
#include <string>

#include "rtc_base/system/rtc_export.h"

namespace cricket {

// Options that can be applied to a VoiceMediaChannel or a VoiceMediaEngine.
// Used to be flags, but that makes it hard to selectively apply options.
// We are moving all of the setting of options to structs like this,
// but some things currently still use flags.
struct RTC_EXPORT AudioOptions {
  AudioOptions();
  ~AudioOptions();
  void SetAll(const AudioOptions& change);

  bool operator==(const AudioOptions& o) const;
  bool operator!=(const AudioOptions& o) const { return !(*this == o); }

  std::string ToString() const;

  // Audio processing that attempts to filter away the output signal from
  // later inbound pickup.
  std::optional<bool> echo_cancellation;
#if defined(WEBRTC_IOS)
  // Forces software echo cancellation on iOS. This is a temporary workaround
  // (until Apple fixes the bug) for a device with non-functioning AEC. May
  // improve performance on that particular device, but will cause unpredictable
  // behavior in all other cases. See http://bugs.webrtc.org/8682.
  std::optional<bool> ios_force_software_aec_HACK;
#endif
  // Audio processing to adjust the sensitivity of the local mic dynamically.
  std::optional<bool> auto_gain_control;
  // Audio processing to filter out background noise.
  std::optional<bool> noise_suppression;
  // Audio processing to remove background noise of lower frequencies.
  std::optional<bool> highpass_filter;
  // Audio processing to swap the left and right channels.
  std::optional<bool> stereo_swapping;
  // Audio receiver jitter buffer (NetEq) max capacity in number of packets.
  std::optional<int> audio_jitter_buffer_max_packets;
  // Audio receiver jitter buffer (NetEq) fast accelerate mode.
  std::optional<bool> audio_jitter_buffer_fast_accelerate;
  // Audio receiver jitter buffer (NetEq) minimum target delay in milliseconds.
  std::optional<int> audio_jitter_buffer_min_delay_ms;
  // Enable audio network adaptor.
  // TODO(webrtc:11717): Remove this API in favor of adaptivePtime in
  // RtpEncodingParameters.
  std::optional<bool> audio_network_adaptor;
  // Config string for audio network adaptor.
  std::optional<std::string> audio_network_adaptor_config;
  // Pre-initialize the ADM for recording when starting to send. Default to
  // true.
  // TODO(webrtc:13566): Remove this option. See issue for details.
  std::optional<bool> init_recording_on_send;
};

}  // namespace cricket

#endif  // API_AUDIO_OPTIONS_H_
