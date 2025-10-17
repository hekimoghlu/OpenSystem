/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 31, 2024.
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
#include "logging/rtc_event_log/events/rtc_event_video_send_stream_config.h"

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "api/rtc_event_log/rtc_event.h"
#include "logging/rtc_event_log/rtc_stream_config.h"

namespace webrtc {

RtcEventVideoSendStreamConfig::RtcEventVideoSendStreamConfig(
    std::unique_ptr<rtclog::StreamConfig> config)
    : config_(std::move(config)) {}

RtcEventVideoSendStreamConfig::RtcEventVideoSendStreamConfig(
    const RtcEventVideoSendStreamConfig& other)
    : RtcEvent(other.timestamp_us_),
      config_(std::make_unique<rtclog::StreamConfig>(*other.config_)) {}

RtcEventVideoSendStreamConfig::~RtcEventVideoSendStreamConfig() = default;

std::unique_ptr<RtcEventVideoSendStreamConfig>
RtcEventVideoSendStreamConfig::Copy() const {
  return absl::WrapUnique<RtcEventVideoSendStreamConfig>(
      new RtcEventVideoSendStreamConfig(*this));
}

}  // namespace webrtc
