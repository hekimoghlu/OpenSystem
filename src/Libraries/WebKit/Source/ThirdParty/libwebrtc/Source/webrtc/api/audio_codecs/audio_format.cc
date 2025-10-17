/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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
#include "api/audio_codecs/audio_format.h"

#include <cstddef>
#include <utility>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "api/rtp_parameters.h"
#include "rtc_base/checks.h"

namespace webrtc {

SdpAudioFormat::SdpAudioFormat(const SdpAudioFormat&) = default;
SdpAudioFormat::SdpAudioFormat(SdpAudioFormat&&) = default;

SdpAudioFormat::SdpAudioFormat(absl::string_view name,
                               int clockrate_hz,
                               size_t num_channels)
    : name(name), clockrate_hz(clockrate_hz), num_channels(num_channels) {}

SdpAudioFormat::SdpAudioFormat(absl::string_view name,
                               int clockrate_hz,
                               size_t num_channels,
                               const CodecParameterMap& param)
    : name(name),
      clockrate_hz(clockrate_hz),
      num_channels(num_channels),
      parameters(param) {}

SdpAudioFormat::SdpAudioFormat(absl::string_view name,
                               int clockrate_hz,
                               size_t num_channels,
                               CodecParameterMap&& param)
    : name(name),
      clockrate_hz(clockrate_hz),
      num_channels(num_channels),
      parameters(std::move(param)) {}

bool SdpAudioFormat::Matches(const SdpAudioFormat& o) const {
  return absl::EqualsIgnoreCase(name, o.name) &&
         clockrate_hz == o.clockrate_hz && num_channels == o.num_channels;
}

SdpAudioFormat::~SdpAudioFormat() = default;
SdpAudioFormat& SdpAudioFormat::operator=(const SdpAudioFormat&) = default;
SdpAudioFormat& SdpAudioFormat::operator=(SdpAudioFormat&&) = default;

bool operator==(const SdpAudioFormat& a, const SdpAudioFormat& b) {
  return absl::EqualsIgnoreCase(a.name, b.name) &&
         a.clockrate_hz == b.clockrate_hz && a.num_channels == b.num_channels &&
         a.parameters == b.parameters;
}

AudioCodecInfo::AudioCodecInfo(int sample_rate_hz,
                               size_t num_channels,
                               int bitrate_bps)
    : AudioCodecInfo(sample_rate_hz,
                     num_channels,
                     bitrate_bps,
                     bitrate_bps,
                     bitrate_bps) {}

AudioCodecInfo::AudioCodecInfo(int sample_rate_hz,
                               size_t num_channels,
                               int default_bitrate_bps,
                               int min_bitrate_bps,
                               int max_bitrate_bps)
    : sample_rate_hz(sample_rate_hz),
      num_channels(num_channels),
      default_bitrate_bps(default_bitrate_bps),
      min_bitrate_bps(min_bitrate_bps),
      max_bitrate_bps(max_bitrate_bps) {
  RTC_DCHECK_GT(sample_rate_hz, 0);
  RTC_DCHECK_GT(num_channels, 0);
  RTC_DCHECK_GE(min_bitrate_bps, 0);
  RTC_DCHECK_LE(min_bitrate_bps, default_bitrate_bps);
  RTC_DCHECK_GE(max_bitrate_bps, default_bitrate_bps);
}

}  // namespace webrtc
