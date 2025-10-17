/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 24, 2025.
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
#include "modules/audio_coding/neteq/tools/initial_packet_inserter_neteq_input.h"

#include <limits>
#include <memory>
#include <utility>

#include "rtc_base/checks.h"

namespace webrtc {
namespace test {

InitialPacketInserterNetEqInput::InitialPacketInserterNetEqInput(
    std::unique_ptr<NetEqInput> source,
    int number_of_initial_packets,
    int sample_rate_hz)
    : source_(std::move(source)),
      packets_to_insert_(number_of_initial_packets),
      sample_rate_hz_(sample_rate_hz) {}

std::optional<int64_t> InitialPacketInserterNetEqInput::NextPacketTime() const {
  return source_->NextPacketTime();
}

std::optional<int64_t> InitialPacketInserterNetEqInput::NextOutputEventTime()
    const {
  return source_->NextOutputEventTime();
}

std::optional<NetEqInput::SetMinimumDelayInfo>
InitialPacketInserterNetEqInput::NextSetMinimumDelayInfo() const {
  return source_->NextSetMinimumDelayInfo();
}

std::unique_ptr<InitialPacketInserterNetEqInput::PacketData>
InitialPacketInserterNetEqInput::PopPacket() {
  if (!first_packet_) {
    first_packet_ = source_->PopPacket();
    if (!first_packet_) {
      // The source has no packets, so we should not insert any dummy packets.
      packets_to_insert_ = 0;
    }
  }
  if (packets_to_insert_ > 0) {
    RTC_CHECK(first_packet_);
    auto dummy_packet = std::unique_ptr<PacketData>(new PacketData());
    dummy_packet->header = first_packet_->header;
    dummy_packet->payload = rtc::Buffer(first_packet_->payload.data(),
                                        first_packet_->payload.size());
    dummy_packet->time_ms = first_packet_->time_ms;
    dummy_packet->header.sequenceNumber -= packets_to_insert_;
    // This assumes 20ms per packet.
    dummy_packet->header.timestamp -=
        20 * sample_rate_hz_ * packets_to_insert_ / 1000;
    packets_to_insert_--;
    return dummy_packet;
  }
  return source_->PopPacket();
}

void InitialPacketInserterNetEqInput::AdvanceSetMinimumDelay() {
  source_->AdvanceSetMinimumDelay();
}

void InitialPacketInserterNetEqInput::AdvanceOutputEvent() {
  source_->AdvanceOutputEvent();
}

bool InitialPacketInserterNetEqInput::ended() const {
  return source_->ended();
}

std::optional<RTPHeader> InitialPacketInserterNetEqInput::NextHeader() const {
  return source_->NextHeader();
}

}  // namespace test
}  // namespace webrtc
