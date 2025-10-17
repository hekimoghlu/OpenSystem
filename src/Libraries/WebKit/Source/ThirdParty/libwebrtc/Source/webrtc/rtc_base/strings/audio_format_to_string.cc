/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 26, 2021.
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
#include "rtc_base/strings/audio_format_to_string.h"

#include <utility>

#include "rtc_base/strings/string_builder.h"

namespace rtc {
std::string ToString(const webrtc::SdpAudioFormat& saf) {
  char sb_buf[1024];
  rtc::SimpleStringBuilder sb(sb_buf);
  sb << "{name: " << saf.name;
  sb << ", clockrate_hz: " << saf.clockrate_hz;
  sb << ", num_channels: " << saf.num_channels;
  sb << ", parameters: {";
  const char* sep = "";
  for (const auto& kv : saf.parameters) {
    sb << sep << kv.first << ": " << kv.second;
    sep = ", ";
  }
  sb << "}}";
  return sb.str();
}
std::string ToString(const webrtc::AudioCodecInfo& aci) {
  char sb_buf[1024];
  rtc::SimpleStringBuilder sb(sb_buf);
  sb << "{sample_rate_hz: " << aci.sample_rate_hz;
  sb << ", num_channels: " << aci.num_channels;
  sb << ", default_bitrate_bps: " << aci.default_bitrate_bps;
  sb << ", min_bitrate_bps: " << aci.min_bitrate_bps;
  sb << ", max_bitrate_bps: " << aci.max_bitrate_bps;
  sb << ", allow_comfort_noise: " << aci.allow_comfort_noise;
  sb << ", supports_network_adaption: " << aci.supports_network_adaption;
  sb << "}";
  return sb.str();
}
std::string ToString(const webrtc::AudioCodecSpec& acs) {
  char sb_buf[1024];
  rtc::SimpleStringBuilder sb(sb_buf);
  sb << "{format: " << ToString(acs.format);
  sb << ", info: " << ToString(acs.info);
  sb << "}";
  return sb.str();
}
}  // namespace rtc
