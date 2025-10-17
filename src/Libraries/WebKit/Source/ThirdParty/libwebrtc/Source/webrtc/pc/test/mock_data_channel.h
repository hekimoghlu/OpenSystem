/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 15, 2022.
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
#ifndef PC_TEST_MOCK_DATA_CHANNEL_H_
#define PC_TEST_MOCK_DATA_CHANNEL_H_

#include <string>
#include <utility>

#include "pc/sctp_data_channel.h"
#include "test/gmock.h"

namespace webrtc {

class MockSctpDataChannel : public SctpDataChannel {
 public:
  MockSctpDataChannel(
      rtc::WeakPtr<SctpDataChannelControllerInterface> controller,
      int id,
      DataState state)
      : MockSctpDataChannel(std::move(controller),
                            id,
                            "MockSctpDataChannel",
                            state,
                            "someProtocol",
                            0,
                            0,
                            0,
                            0) {}
  MockSctpDataChannel(
      rtc::WeakPtr<SctpDataChannelControllerInterface> controller,
      int id,
      const std::string& label,
      DataState state,
      const std::string& protocol,
      uint32_t messages_sent,
      uint64_t bytes_sent,
      uint32_t messages_received,
      uint64_t bytes_received,
      const InternalDataChannelInit& config = InternalDataChannelInit(),
      rtc::Thread* signaling_thread = rtc::Thread::Current(),
      rtc::Thread* network_thread = rtc::Thread::Current())
      : SctpDataChannel(config,
                        std::move(controller),
                        label,
                        false,
                        signaling_thread,
                        network_thread) {
    EXPECT_CALL(*this, id()).WillRepeatedly(::testing::Return(id));
    EXPECT_CALL(*this, state()).WillRepeatedly(::testing::Return(state));
    EXPECT_CALL(*this, protocol()).WillRepeatedly(::testing::Return(protocol));
    EXPECT_CALL(*this, messages_sent())
        .WillRepeatedly(::testing::Return(messages_sent));
    EXPECT_CALL(*this, bytes_sent())
        .WillRepeatedly(::testing::Return(bytes_sent));
    EXPECT_CALL(*this, messages_received())
        .WillRepeatedly(::testing::Return(messages_received));
    EXPECT_CALL(*this, bytes_received())
        .WillRepeatedly(::testing::Return(bytes_received));
  }
  MOCK_METHOD(int, id, (), (const, override));
  MOCK_METHOD(DataState, state, (), (const, override));
  MOCK_METHOD(std::string, protocol, (), (const, override));
  MOCK_METHOD(uint32_t, messages_sent, (), (const, override));
  MOCK_METHOD(uint64_t, bytes_sent, (), (const, override));
  MOCK_METHOD(uint32_t, messages_received, (), (const, override));
  MOCK_METHOD(uint64_t, bytes_received, (), (const, override));
};

}  // namespace webrtc

#endif  // PC_TEST_MOCK_DATA_CHANNEL_H_
