/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 12, 2024.
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
#ifndef LOGGING_RTC_EVENT_LOG_DEPENDENCY_DESCRIPTOR_ENCODER_DECODER_H_
#define LOGGING_RTC_EVENT_LOG_DEPENDENCY_DESCRIPTOR_ENCODER_DECODER_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "api/array_view.h"
#include "logging/rtc_event_log/events/rtc_event_log_parse_status.h"
#include "logging/rtc_event_log/rtc_event_log2_proto_include.h"

namespace webrtc {

class RtcEventLogDependencyDescriptorEncoderDecoder {
 public:
  static std::optional<rtclog2::DependencyDescriptorsWireInfo> Encode(
      const std::vector<rtc::ArrayView<const uint8_t>>& raw_dd_data);
  static RtcEventLogParseStatusOr<std::vector<std::vector<uint8_t>>> Decode(
      const rtclog2::DependencyDescriptorsWireInfo& dd_wire_info,
      size_t num_packets);
};

}  // namespace webrtc

#endif  //  LOGGING_RTC_EVENT_LOG_DEPENDENCY_DESCRIPTOR_ENCODER_DECODER_H_
