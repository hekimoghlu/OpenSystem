/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 3, 2025.
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
#include "modules/audio_coding/neteq/tools/neteq_input.h"

#include "rtc_base/strings/string_builder.h"

namespace webrtc {
namespace test {

NetEqInput::PacketData::PacketData() = default;
NetEqInput::PacketData::~PacketData() = default;

std::string NetEqInput::PacketData::ToString() const {
  rtc::StringBuilder ss;
  ss << "{"
        "time_ms: "
     << static_cast<int64_t>(time_ms)
     << ", "
        "header: {"
        "pt: "
     << static_cast<int>(header.payloadType)
     << ", "
        "sn: "
     << header.sequenceNumber
     << ", "
        "ts: "
     << header.timestamp
     << ", "
        "ssrc: "
     << header.ssrc
     << "}, "
        "payload bytes: "
     << payload.size() << "}";
  return ss.Release();
}

TimeLimitedNetEqInput::TimeLimitedNetEqInput(std::unique_ptr<NetEqInput> input,
                                             int64_t duration_ms)
    : input_(std::move(input)),
      start_time_ms_(input_->NextEventTime()),
      duration_ms_(duration_ms) {}

TimeLimitedNetEqInput::~TimeLimitedNetEqInput() = default;

std::optional<int64_t> TimeLimitedNetEqInput::NextPacketTime() const {
  return ended_ ? std::nullopt : input_->NextPacketTime();
}

std::optional<int64_t> TimeLimitedNetEqInput::NextOutputEventTime() const {
  return ended_ ? std::nullopt : input_->NextOutputEventTime();
}

std::optional<NetEqInput::SetMinimumDelayInfo>
TimeLimitedNetEqInput::NextSetMinimumDelayInfo() const {
  return ended_ ? std::nullopt : input_->NextSetMinimumDelayInfo();
}

std::unique_ptr<NetEqInput::PacketData> TimeLimitedNetEqInput::PopPacket() {
  if (ended_) {
    return std::unique_ptr<PacketData>();
  }
  auto packet = input_->PopPacket();
  MaybeSetEnded();
  return packet;
}

void TimeLimitedNetEqInput::AdvanceOutputEvent() {
  if (!ended_) {
    input_->AdvanceOutputEvent();
    MaybeSetEnded();
  }
}

void TimeLimitedNetEqInput::AdvanceSetMinimumDelay() {
  if (!ended_) {
    input_->AdvanceSetMinimumDelay();
    MaybeSetEnded();
  }
}

bool TimeLimitedNetEqInput::ended() const {
  return ended_ || input_->ended();
}

std::optional<RTPHeader> TimeLimitedNetEqInput::NextHeader() const {
  return ended_ ? std::nullopt : input_->NextHeader();
}

void TimeLimitedNetEqInput::MaybeSetEnded() {
  if (NextEventTime() && start_time_ms_ &&
      *NextEventTime() - *start_time_ms_ > duration_ms_) {
    ended_ = true;
  }
}

}  // namespace test
}  // namespace webrtc
