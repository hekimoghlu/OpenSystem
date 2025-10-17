/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 1, 2021.
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
#include "call/audio_send_stream.h"

#include <stddef.h>

#include <string>

#include "api/audio_codecs/audio_format.h"
#include "api/call/transport.h"
#include "rtc_base/string_encode.h"
#include "rtc_base/strings/audio_format_to_string.h"
#include "rtc_base/strings/string_builder.h"

namespace webrtc {

AudioSendStream::Stats::Stats() = default;
AudioSendStream::Stats::~Stats() = default;

AudioSendStream::Config::Config(Transport* send_transport)
    : send_transport(send_transport) {}

AudioSendStream::Config::~Config() = default;

std::string AudioSendStream::Config::ToString() const {
  rtc::StringBuilder ss;
  ss << "{rtp: " << rtp.ToString();
  ss << ", rtcp_report_interval_ms: " << rtcp_report_interval_ms;
  ss << ", send_transport: " << (send_transport ? "(Transport)" : "null");
  ss << ", min_bitrate_bps: " << min_bitrate_bps;
  ss << ", max_bitrate_bps: " << max_bitrate_bps;
  ss << ", has audio_network_adaptor_config: "
     << (audio_network_adaptor_config ? "true" : "false");
  ss << ", has_dscp: " << (has_dscp ? "true" : "false");
  ss << ", send_codec_spec: "
     << (send_codec_spec ? send_codec_spec->ToString() : "<unset>");
  ss << "}";
  return ss.Release();
}

AudioSendStream::Config::Rtp::Rtp() = default;

AudioSendStream::Config::Rtp::~Rtp() = default;

std::string AudioSendStream::Config::Rtp::ToString() const {
  char buf[1024];
  rtc::SimpleStringBuilder ss(buf);
  ss << "{ssrc: " << ssrc;
  if (!rid.empty()) {
    ss << ", rid: " << rid;
  }
  if (!mid.empty()) {
    ss << ", mid: " << mid;
  }
  ss << ", extmap-allow-mixed: " << (extmap_allow_mixed ? "true" : "false");
  ss << ", extensions: [";
  for (size_t i = 0; i < extensions.size(); ++i) {
    ss << extensions[i].ToString();
    if (i != extensions.size() - 1) {
      ss << ", ";
    }
  }
  ss << ']';
  ss << ", c_name: " << c_name;
  ss << '}';
  return ss.str();
}

AudioSendStream::Config::SendCodecSpec::SendCodecSpec(
    int payload_type,
    const SdpAudioFormat& format)
    : payload_type(payload_type), format(format) {}
AudioSendStream::Config::SendCodecSpec::~SendCodecSpec() = default;

std::string AudioSendStream::Config::SendCodecSpec::ToString() const {
  char buf[1024];
  rtc::SimpleStringBuilder ss(buf);
  ss << "{nack_enabled: " << (nack_enabled ? "true" : "false");
  ss << ", transport_cc_enabled: " << (transport_cc_enabled ? "true" : "false");
  ss << ", enable_non_sender_rtt: "
     << (enable_non_sender_rtt ? "true" : "false");
  ss << ", cng_payload_type: "
     << (cng_payload_type ? rtc::ToString(*cng_payload_type) : "<unset>");
  ss << ", red_payload_type: "
     << (red_payload_type ? rtc::ToString(*red_payload_type) : "<unset>");
  ss << ", payload_type: " << payload_type;
  ss << ", format: " << rtc::ToString(format);
  ss << '}';
  return ss.str();
}

bool AudioSendStream::Config::SendCodecSpec::operator==(
    const AudioSendStream::Config::SendCodecSpec& rhs) const {
  if (nack_enabled == rhs.nack_enabled &&
      transport_cc_enabled == rhs.transport_cc_enabled &&
      enable_non_sender_rtt == rhs.enable_non_sender_rtt &&
      cng_payload_type == rhs.cng_payload_type &&
      red_payload_type == rhs.red_payload_type &&
      payload_type == rhs.payload_type && format == rhs.format &&
      target_bitrate_bps == rhs.target_bitrate_bps) {
    return true;
  }
  return false;
}
}  // namespace webrtc
