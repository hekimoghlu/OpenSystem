/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_MOCK_MOCK_PACKET_BUFFER_H_
#define MODULES_AUDIO_CODING_NETEQ_MOCK_MOCK_PACKET_BUFFER_H_

#include "modules/audio_coding/neteq/packet_buffer.h"
#include "test/gmock.h"

namespace webrtc {

class MockPacketBuffer : public PacketBuffer {
 public:
  MockPacketBuffer(size_t max_number_of_packets,
                   const TickTimer* tick_timer,
                   StatisticsCalculator* stats)
      : PacketBuffer(max_number_of_packets, tick_timer, stats) {}
  ~MockPacketBuffer() override { Die(); }
  MOCK_METHOD(void, Die, ());
  MOCK_METHOD(void, Flush, (), (override));
  MOCK_METHOD(bool, Empty, (), (const, override));
  MOCK_METHOD(int, InsertPacket, (Packet && packet), (override));
  MOCK_METHOD(int,
              NextTimestamp,
              (uint32_t * next_timestamp),
              (const, override));
  MOCK_METHOD(int,
              NextHigherTimestamp,
              (uint32_t timestamp, uint32_t* next_timestamp),
              (const, override));
  MOCK_METHOD(const Packet*, PeekNextPacket, (), (const, override));
  MOCK_METHOD(std::optional<Packet>, GetNextPacket, (), (override));
  MOCK_METHOD(int, DiscardNextPacket, (), (override));
  MOCK_METHOD(void,
              DiscardOldPackets,
              (uint32_t timestamp_limit, uint32_t horizon_samples),
              (override));
  MOCK_METHOD(void,
              DiscardAllOldPackets,
              (uint32_t timestamp_limit),
              (override));
  MOCK_METHOD(size_t, NumPacketsInBuffer, (), (const, override));
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_MOCK_MOCK_PACKET_BUFFER_H_
