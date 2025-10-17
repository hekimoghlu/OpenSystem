/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 4, 2024.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TOOLS_CONSTANT_PCM_PACKET_SOURCE_H_
#define MODULES_AUDIO_CODING_NETEQ_TOOLS_CONSTANT_PCM_PACKET_SOURCE_H_

#include <stdio.h>

#include <string>

#include "modules/audio_coding/neteq/tools/packet_source.h"

namespace webrtc {
namespace test {

// This class implements a packet source that delivers PCM16b encoded packets
// with a constant sample value. The payload length, constant sample value,
// sample rate, and payload type are all set in the constructor.
class ConstantPcmPacketSource : public PacketSource {
 public:
  ConstantPcmPacketSource(size_t payload_len_samples,
                          int16_t sample_value,
                          int sample_rate_hz,
                          int payload_type);

  ConstantPcmPacketSource(const ConstantPcmPacketSource&) = delete;
  ConstantPcmPacketSource& operator=(const ConstantPcmPacketSource&) = delete;

  std::unique_ptr<Packet> NextPacket() override;

 private:
  void WriteHeader(uint8_t* packet_memory);

  const size_t kHeaderLenBytes = 12;
  const size_t payload_len_samples_;
  const size_t packet_len_bytes_;
  uint8_t encoded_sample_[2];
  const int samples_per_ms_;
  double next_arrival_time_ms_;
  const int payload_type_;
  uint16_t seq_number_;
  uint32_t timestamp_;
  const uint32_t payload_ssrc_;
};

}  // namespace test
}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_TOOLS_CONSTANT_PCM_PACKET_SOURCE_H_
