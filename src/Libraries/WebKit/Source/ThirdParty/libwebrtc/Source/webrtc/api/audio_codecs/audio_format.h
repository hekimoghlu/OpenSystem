/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 2, 2022.
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
#ifndef API_AUDIO_CODECS_AUDIO_FORMAT_H_
#define API_AUDIO_CODECS_AUDIO_FORMAT_H_

#include <stddef.h>

#include <map>
#include <string>

#include "absl/strings/string_view.h"
#include "api/rtp_parameters.h"
#include "rtc_base/checks.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// SDP specification for a single audio codec.
struct RTC_EXPORT SdpAudioFormat {
  using Parameters [[deprecated("Use webrtc::CodecParameterMap")]] =
      std::map<std::string, std::string>;

  SdpAudioFormat(const SdpAudioFormat&);
  SdpAudioFormat(SdpAudioFormat&&);
  SdpAudioFormat(absl::string_view name, int clockrate_hz, size_t num_channels);
  SdpAudioFormat(absl::string_view name,
                 int clockrate_hz,
                 size_t num_channels,
                 const CodecParameterMap& param);
  SdpAudioFormat(absl::string_view name,
                 int clockrate_hz,
                 size_t num_channels,
                 CodecParameterMap&& param);
  ~SdpAudioFormat();

  // Returns true if this format is compatible with `o`. In SDP terminology:
  // would it represent the same codec between an offer and an answer? As
  // opposed to operator==, this method disregards codec parameters.
  bool Matches(const SdpAudioFormat& o) const;

  SdpAudioFormat& operator=(const SdpAudioFormat&);
  SdpAudioFormat& operator=(SdpAudioFormat&&);

  friend bool operator==(const SdpAudioFormat& a, const SdpAudioFormat& b);
  friend bool operator!=(const SdpAudioFormat& a, const SdpAudioFormat& b) {
    return !(a == b);
  }

  std::string name;
  int clockrate_hz;
  size_t num_channels;
  CodecParameterMap parameters;
};

// Information about how an audio format is treated by the codec implementation.
// Contains basic information, such as sample rate and number of channels, which
// isn't uniformly presented by SDP. Also contains flags indicating support for
// integrating with other parts of WebRTC, like external VAD and comfort noise
// level calculation.
//
// To avoid API breakage, and make the code clearer, AudioCodecInfo should not
// be directly initializable with any flags indicating optional support. If it
// were, these initializers would break any time a new flag was added. It's also
// more difficult to understand:
//   AudioCodecInfo info{16000, 1, 32000, true, false, false, true, true};
// than
//   AudioCodecInfo info(16000, 1, 32000);
//   info.allow_comfort_noise = true;
//   info.future_flag_b = true;
//   info.future_flag_c = true;
struct AudioCodecInfo {
  AudioCodecInfo(int sample_rate_hz, size_t num_channels, int bitrate_bps);
  AudioCodecInfo(int sample_rate_hz,
                 size_t num_channels,
                 int default_bitrate_bps,
                 int min_bitrate_bps,
                 int max_bitrate_bps);
  AudioCodecInfo(const AudioCodecInfo& b) = default;
  ~AudioCodecInfo() = default;

  bool operator==(const AudioCodecInfo& b) const {
    return sample_rate_hz == b.sample_rate_hz &&
           num_channels == b.num_channels &&
           default_bitrate_bps == b.default_bitrate_bps &&
           min_bitrate_bps == b.min_bitrate_bps &&
           max_bitrate_bps == b.max_bitrate_bps &&
           allow_comfort_noise == b.allow_comfort_noise &&
           supports_network_adaption == b.supports_network_adaption;
  }

  bool operator!=(const AudioCodecInfo& b) const { return !(*this == b); }

  bool HasFixedBitrate() const {
    RTC_DCHECK_GE(min_bitrate_bps, 0);
    RTC_DCHECK_LE(min_bitrate_bps, default_bitrate_bps);
    RTC_DCHECK_GE(max_bitrate_bps, default_bitrate_bps);
    return min_bitrate_bps == max_bitrate_bps;
  }

  int sample_rate_hz;
  size_t num_channels;
  int default_bitrate_bps;
  int min_bitrate_bps;
  int max_bitrate_bps;

  bool allow_comfort_noise = true;  // This codec can be used with an external
                                    // comfort noise generator.
  bool supports_network_adaption = false;  // This codec can adapt to varying
                                           // network conditions.
};

// AudioCodecSpec ties an audio format to specific information about the codec
// and its implementation.
struct AudioCodecSpec {
  bool operator==(const AudioCodecSpec& b) const {
    return format == b.format && info == b.info;
  }

  bool operator!=(const AudioCodecSpec& b) const { return !(*this == b); }

  SdpAudioFormat format;
  AudioCodecInfo info;
};

}  // namespace webrtc

#endif  // API_AUDIO_CODECS_AUDIO_FORMAT_H_
