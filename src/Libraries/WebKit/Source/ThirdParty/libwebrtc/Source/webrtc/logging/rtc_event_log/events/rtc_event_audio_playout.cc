/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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
#include "logging/rtc_event_log/events/rtc_event_audio_playout.h"

#include <cstdint>
#include <memory>

#include "absl/memory/memory.h"
#include "api/rtc_event_log/rtc_event.h"
#include "logging/rtc_event_log/events/rtc_event_definition.h"

namespace webrtc {

constexpr RtcEventDefinition<RtcEventAudioPlayout,
                             LoggedAudioPlayoutEvent,
                             uint32_t>
    RtcEventAudioPlayout::definition_;

RtcEventAudioPlayout::RtcEventAudioPlayout(uint32_t ssrc) : ssrc_(ssrc) {}

RtcEventAudioPlayout::RtcEventAudioPlayout(const RtcEventAudioPlayout& other)
    : RtcEvent(other.timestamp_us_), ssrc_(other.ssrc_) {}

std::unique_ptr<RtcEventAudioPlayout> RtcEventAudioPlayout::Copy() const {
  return absl::WrapUnique<RtcEventAudioPlayout>(
      new RtcEventAudioPlayout(*this));
}

}  // namespace webrtc
