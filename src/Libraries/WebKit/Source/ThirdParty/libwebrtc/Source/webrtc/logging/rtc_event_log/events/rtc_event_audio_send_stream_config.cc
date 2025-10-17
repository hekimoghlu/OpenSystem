/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 13, 2023.
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
#include "logging/rtc_event_log/events/rtc_event_audio_send_stream_config.h"

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "api/rtc_event_log/rtc_event.h"
#include "logging/rtc_event_log/rtc_stream_config.h"
#include "rtc_base/checks.h"

namespace webrtc {

RtcEventAudioSendStreamConfig::RtcEventAudioSendStreamConfig(
    std::unique_ptr<rtclog::StreamConfig> config)
    : config_(std::move(config)) {
  RTC_DCHECK(config_);
}

RtcEventAudioSendStreamConfig::RtcEventAudioSendStreamConfig(
    const RtcEventAudioSendStreamConfig& other)
    : RtcEvent(other.timestamp_us_),
      config_(std::make_unique<rtclog::StreamConfig>(*other.config_)) {}

RtcEventAudioSendStreamConfig::~RtcEventAudioSendStreamConfig() = default;

std::unique_ptr<RtcEventAudioSendStreamConfig>
RtcEventAudioSendStreamConfig::Copy() const {
  return absl::WrapUnique<RtcEventAudioSendStreamConfig>(
      new RtcEventAudioSendStreamConfig(*this));
}

}  // namespace webrtc
