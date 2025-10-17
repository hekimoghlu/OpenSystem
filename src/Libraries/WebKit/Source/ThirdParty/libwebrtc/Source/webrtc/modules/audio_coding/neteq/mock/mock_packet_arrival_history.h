/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 19, 2023.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_MOCK_MOCK_PACKET_ARRIVAL_HISTORY_H_
#define MODULES_AUDIO_CODING_NETEQ_MOCK_MOCK_PACKET_ARRIVAL_HISTORY_H_

#include "api/neteq/tick_timer.h"
#include "modules/audio_coding/neteq/packet_arrival_history.h"
#include "test/gmock.h"

namespace webrtc {

class MockPacketArrivalHistory : public PacketArrivalHistory {
 public:
  MockPacketArrivalHistory(const TickTimer* tick_timer)
      : PacketArrivalHistory(tick_timer, 0) {}

  MOCK_METHOD(int, GetDelayMs, (uint32_t rtp_timestamp), (const, override));
  MOCK_METHOD(int, GetMaxDelayMs, (), (const, override));
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_NETEQ_MOCK_MOCK_PACKET_ARRIVAL_HISTORY_H_
