/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 6, 2023.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_MOCK_MOCK_NETEQ_CONTROLLER_H_
#define MODULES_AUDIO_CODING_NETEQ_MOCK_MOCK_NETEQ_CONTROLLER_H_

#include "api/neteq/neteq_controller.h"
#include "test/gmock.h"

namespace webrtc {

class MockNetEqController : public NetEqController {
 public:
  MockNetEqController() = default;
  ~MockNetEqController() override { Die(); }
  MOCK_METHOD(void, Die, ());
  MOCK_METHOD(void, Reset, (), (override));
  MOCK_METHOD(void, SoftReset, (), (override));
  MOCK_METHOD(NetEq::Operation,
              GetDecision,
              (const NetEqStatus& neteq_status, bool* reset_decoder),
              (override));
  MOCK_METHOD(void, RegisterEmptyPacket, (), (override));
  MOCK_METHOD(void,
              SetSampleRate,
              (int fs_hz, size_t output_size_samples),
              (override));
  MOCK_METHOD(bool, SetMaximumDelay, (int delay_ms), (override));
  MOCK_METHOD(bool, SetMinimumDelay, (int delay_ms), (override));
  MOCK_METHOD(bool, SetBaseMinimumDelay, (int delay_ms), (override));
  MOCK_METHOD(int, GetBaseMinimumDelay, (), (const, override));
  MOCK_METHOD(void, ExpandDecision, (NetEq::Operation operation), (override));
  MOCK_METHOD(void, AddSampleMemory, (int32_t value), (override));
  MOCK_METHOD(int, TargetLevelMs, (), (const, override));
  MOCK_METHOD(std::optional<int>,
              PacketArrived,
              (int fs_hz,
               bool should_update_stats,
               const PacketArrivedInfo& info),
              (override));
  MOCK_METHOD(void, NotifyMutedState, (), (override));
  MOCK_METHOD(bool, PeakFound, (), (const, override));
  MOCK_METHOD(int, GetFilteredBufferLevel, (), (const, override));
  MOCK_METHOD(void, set_sample_memory, (int32_t value), (override));
  MOCK_METHOD(size_t, noise_fast_forward, (), (const, override));
  MOCK_METHOD(size_t, packet_length_samples, (), (const, override));
  MOCK_METHOD(void, set_packet_length_samples, (size_t value), (override));
  MOCK_METHOD(void, set_prev_time_scale, (bool value), (override));
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_MOCK_MOCK_NETEQ_CONTROLLER_H_
