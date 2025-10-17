/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 18, 2024.
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
#include "logging/rtc_event_log/rtc_stream_config.h"

#include "absl/strings/string_view.h"

namespace webrtc {
namespace rtclog {

StreamConfig::StreamConfig() {}

StreamConfig::~StreamConfig() {}

StreamConfig::StreamConfig(const StreamConfig& other) = default;

bool StreamConfig::operator==(const StreamConfig& other) const {
  return local_ssrc == other.local_ssrc && remote_ssrc == other.remote_ssrc &&
         rtx_ssrc == other.rtx_ssrc && rsid == other.rsid &&
         remb == other.remb && rtcp_mode == other.rtcp_mode &&
         rtp_extensions == other.rtp_extensions && codecs == other.codecs;
}

bool StreamConfig::operator!=(const StreamConfig& other) const {
  return !(*this == other);
}

StreamConfig::Codec::Codec(absl::string_view payload_name,
                           int payload_type,
                           int rtx_payload_type)
    : payload_name(payload_name),
      payload_type(payload_type),
      rtx_payload_type(rtx_payload_type) {}

bool StreamConfig::Codec::operator==(const Codec& other) const {
  return payload_name == other.payload_name &&
         payload_type == other.payload_type &&
         rtx_payload_type == other.rtx_payload_type;
}

}  // namespace rtclog
}  // namespace webrtc
