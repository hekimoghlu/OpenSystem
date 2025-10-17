/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TOOLS_INITIAL_PACKET_INSERTER_NETEQ_INPUT_H_
#define MODULES_AUDIO_CODING_NETEQ_TOOLS_INITIAL_PACKET_INSERTER_NETEQ_INPUT_H_

#include <map>
#include <memory>
#include <string>

#include "modules/audio_coding/neteq/tools/neteq_input.h"

namespace webrtc {
namespace test {

// Wrapper class that can insert a number of packets at the start of the
// simulation.
class InitialPacketInserterNetEqInput final : public NetEqInput {
 public:
  InitialPacketInserterNetEqInput(std::unique_ptr<NetEqInput> source,
                                  int number_of_initial_packets,
                                  int sample_rate_hz);
  std::optional<int64_t> NextPacketTime() const override;
  std::optional<int64_t> NextOutputEventTime() const override;
  std::optional<SetMinimumDelayInfo> NextSetMinimumDelayInfo() const override;
  std::unique_ptr<PacketData> PopPacket() override;
  void AdvanceOutputEvent() override;
  void AdvanceSetMinimumDelay() override;
  bool ended() const override;
  std::optional<RTPHeader> NextHeader() const override;

 private:
  const std::unique_ptr<NetEqInput> source_;
  int packets_to_insert_;
  const int sample_rate_hz_;
  std::unique_ptr<PacketData> first_packet_;
};

}  // namespace test
}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_TOOLS_INITIAL_PACKET_INSERTER_NETEQ_INPUT_H_
