/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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
#ifndef NET_DCSCTP_TX_MOCK_SEND_QUEUE_H_
#define NET_DCSCTP_TX_MOCK_SEND_QUEUE_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "api/array_view.h"
#include "api/units/timestamp.h"
#include "net/dcsctp/tx/send_queue.h"
#include "test/gmock.h"

namespace dcsctp {

class MockSendQueue : public SendQueue {
 public:
  MockSendQueue() {
    ON_CALL(*this, Produce)
        .WillByDefault([](webrtc::Timestamp now, size_t max_size) {
          return std::nullopt;
        });
  }

  MOCK_METHOD(std::optional<SendQueue::DataToSend>,
              Produce,
              (webrtc::Timestamp now, size_t max_size),
              (override));
  MOCK_METHOD(bool,
              Discard,
              (StreamID stream_id, OutgoingMessageId message_id),
              (override));
  MOCK_METHOD(void, PrepareResetStream, (StreamID stream_id), (override));
  MOCK_METHOD(bool, HasStreamsReadyToBeReset, (), (const, override));
  MOCK_METHOD(std::vector<StreamID>, GetStreamsReadyToBeReset, (), (override));
  MOCK_METHOD(void, CommitResetStreams, (), (override));
  MOCK_METHOD(void, RollbackResetStreams, (), (override));
  MOCK_METHOD(void, Reset, (), (override));
  MOCK_METHOD(size_t, buffered_amount, (StreamID stream_id), (const, override));
  MOCK_METHOD(size_t, total_buffered_amount, (), (const, override));
  MOCK_METHOD(size_t,
              buffered_amount_low_threshold,
              (StreamID stream_id),
              (const, override));
  MOCK_METHOD(void,
              SetBufferedAmountLowThreshold,
              (StreamID stream_id, size_t bytes),
              (override));
  MOCK_METHOD(void, EnableMessageInterleaving, (bool enabled), (override));
};

}  // namespace dcsctp

#endif  // NET_DCSCTP_TX_MOCK_SEND_QUEUE_H_
