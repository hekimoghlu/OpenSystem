/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_REPLACEMENT_INPUT_H_
#define MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_REPLACEMENT_INPUT_H_

#include <memory>
#include <set>

#include "modules/audio_coding/neteq/tools/neteq_input.h"

namespace webrtc {
namespace test {

// This class converts the packets from a NetEqInput to fake encodings to be
// decoded by a FakeDecodeFromFile decoder.
class NetEqReplacementInput : public NetEqInput {
 public:
  NetEqReplacementInput(std::unique_ptr<NetEqInput> source,
                        uint8_t replacement_payload_type,
                        const std::set<uint8_t>& comfort_noise_types,
                        const std::set<uint8_t>& forbidden_types);

  std::optional<int64_t> NextPacketTime() const override;
  std::optional<int64_t> NextOutputEventTime() const override;
  std::optional<SetMinimumDelayInfo> NextSetMinimumDelayInfo() const override;
  std::unique_ptr<PacketData> PopPacket() override;
  void AdvanceOutputEvent() override;
  void AdvanceSetMinimumDelay() override;
  bool ended() const override;
  std::optional<RTPHeader> NextHeader() const override;

 private:
  void ReplacePacket();

  std::unique_ptr<NetEqInput> source_;
  const uint8_t replacement_payload_type_;
  const std::set<uint8_t> comfort_noise_types_;
  const std::set<uint8_t> forbidden_types_;
  std::unique_ptr<PacketData> packet_;         // The next packet to deliver.
  uint32_t last_frame_size_timestamps_ = 960;  // Initial guess: 20 ms @ 48 kHz.
};

}  // namespace test
}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_REPLACEMENT_INPUT_H_
