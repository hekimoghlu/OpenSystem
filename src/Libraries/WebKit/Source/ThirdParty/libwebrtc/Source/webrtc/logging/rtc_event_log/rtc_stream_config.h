/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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
#ifndef LOGGING_RTC_EVENT_LOG_RTC_STREAM_CONFIG_H_
#define LOGGING_RTC_EVENT_LOG_RTC_STREAM_CONFIG_H_

#include <stdint.h>

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/rtp_headers.h"
#include "api/rtp_parameters.h"

namespace webrtc {
namespace rtclog {

struct StreamConfig {
  StreamConfig();
  StreamConfig(const StreamConfig& other);
  ~StreamConfig();

  bool operator==(const StreamConfig& other) const;
  bool operator!=(const StreamConfig& other) const;

  uint32_t local_ssrc = 0;
  uint32_t remote_ssrc = 0;
  uint32_t rtx_ssrc = 0;
  std::string rsid;

  bool remb = false;
  std::vector<RtpExtension> rtp_extensions;

  RtcpMode rtcp_mode = RtcpMode::kReducedSize;

  struct Codec {
    Codec(absl::string_view payload_name,
          int payload_type,
          int rtx_payload_type);

    bool operator==(const Codec& other) const;

    std::string payload_name;
    int payload_type;
    int rtx_payload_type;
  };

  std::vector<Codec> codecs;
};

}  // namespace rtclog
}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_RTC_STREAM_CONFIG_H_
