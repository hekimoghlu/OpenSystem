/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 10, 2023.
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
#include "api/neteq/neteq.h"

#include "rtc_base/strings/string_builder.h"

namespace webrtc {

NetEq::Config::Config() = default;
NetEq::Config::Config(const Config&) = default;
NetEq::Config::Config(Config&&) = default;
NetEq::Config::~Config() = default;
NetEq::Config& NetEq::Config::operator=(const Config&) = default;
NetEq::Config& NetEq::Config::operator=(Config&&) = default;

std::string NetEq::Config::ToString() const {
  char buf[1024];
  rtc::SimpleStringBuilder ss(buf);
  ss << "sample_rate_hz=" << sample_rate_hz
     << ", max_packets_in_buffer=" << max_packets_in_buffer
     << ", min_delay_ms=" << min_delay_ms << ", enable_fast_accelerate="
     << (enable_fast_accelerate ? "true" : "false")
     << ", enable_muted_state=" << (enable_muted_state ? "true" : "false")
     << ", enable_rtx_handling=" << (enable_rtx_handling ? "true" : "false");
  return ss.str();
}

}  // namespace webrtc
