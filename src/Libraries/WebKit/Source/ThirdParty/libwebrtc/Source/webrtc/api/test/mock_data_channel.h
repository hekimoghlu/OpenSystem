/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 27, 2022.
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
#ifndef API_TEST_MOCK_DATA_CHANNEL_H_
#define API_TEST_MOCK_DATA_CHANNEL_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/functional/any_invocable.h"
#include "api/data_channel_interface.h"
#include "api/priority.h"
#include "api/rtc_error.h"
#include "api/scoped_refptr.h"
#include "rtc_base/ref_counted_object.h"
#include "test/gmock.h"

namespace webrtc {

class MockDataChannelInterface
    : public rtc::RefCountedObject<webrtc::DataChannelInterface> {
 public:
  static rtc::scoped_refptr<MockDataChannelInterface> Create() {
    return rtc::scoped_refptr<MockDataChannelInterface>(
        new MockDataChannelInterface());
  }

  MOCK_METHOD(void,
              RegisterObserver,
              (DataChannelObserver * observer),
              (override));
  MOCK_METHOD(void, UnregisterObserver, (), (override));
  MOCK_METHOD(std::string, label, (), (const, override));
  MOCK_METHOD(bool, reliable, (), (const, override));
  MOCK_METHOD(bool, ordered, (), (const, override));
  MOCK_METHOD(uint16_t, maxRetransmitTime, (), (const, override));
  MOCK_METHOD(uint16_t, maxRetransmits, (), (const, override));
  MOCK_METHOD(std::optional<int>, maxRetransmitsOpt, (), (const, override));
  MOCK_METHOD(std::optional<int>, maxPacketLifeTime, (), (const, override));
  MOCK_METHOD(std::string, protocol, (), (const, override));
  MOCK_METHOD(bool, negotiated, (), (const, override));
  MOCK_METHOD(int, id, (), (const, override));
  MOCK_METHOD(PriorityValue, priority, (), (const, override));
  MOCK_METHOD(DataState, state, (), (const, override));
  MOCK_METHOD(RTCError, error, (), (const, override));
  MOCK_METHOD(uint32_t, messages_sent, (), (const, override));
  MOCK_METHOD(uint64_t, bytes_sent, (), (const, override));
  MOCK_METHOD(uint32_t, messages_received, (), (const, override));
  MOCK_METHOD(uint64_t, bytes_received, (), (const, override));
  MOCK_METHOD(uint64_t, buffered_amount, (), (const, override));
  MOCK_METHOD(void, Close, (), (override));
  MOCK_METHOD(bool, Send, (const DataBuffer& buffer), (override));
  MOCK_METHOD(void,
              SendAsync,
              (DataBuffer buffer,
               absl::AnyInvocable<void(RTCError) &&> on_complete),
              (override));

 protected:
  MockDataChannelInterface() = default;
};

}  // namespace webrtc

#endif  // API_TEST_MOCK_DATA_CHANNEL_H_
