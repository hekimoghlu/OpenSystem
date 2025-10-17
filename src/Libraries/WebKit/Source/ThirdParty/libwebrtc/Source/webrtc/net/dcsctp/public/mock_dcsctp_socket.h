/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 20, 2025.
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
#ifndef NET_DCSCTP_PUBLIC_MOCK_DCSCTP_SOCKET_H_
#define NET_DCSCTP_PUBLIC_MOCK_DCSCTP_SOCKET_H_

#include <vector>

#include "net/dcsctp/public/dcsctp_socket.h"
#include "test/gmock.h"

namespace dcsctp {

class MockDcSctpSocket : public DcSctpSocketInterface {
 public:
  MOCK_METHOD(void,
              ReceivePacket,
              (rtc::ArrayView<const uint8_t> data),
              (override));

  MOCK_METHOD(void, HandleTimeout, (TimeoutID timeout_id), (override));

  MOCK_METHOD(void, Connect, (), (override));

  MOCK_METHOD(void,
              RestoreFromState,
              (const DcSctpSocketHandoverState&),
              (override));

  MOCK_METHOD(void, Shutdown, (), (override));

  MOCK_METHOD(void, Close, (), (override));

  MOCK_METHOD(SocketState, state, (), (const, override));

  MOCK_METHOD(const DcSctpOptions&, options, (), (const, override));

  MOCK_METHOD(void, SetMaxMessageSize, (size_t max_message_size), (override));

  MOCK_METHOD(void,
              SetStreamPriority,
              (StreamID stream_id, StreamPriority priority),
              (override));

  MOCK_METHOD(StreamPriority,
              GetStreamPriority,
              (StreamID stream_id),
              (const, override));

  MOCK_METHOD(SendStatus,
              Send,
              (DcSctpMessage message, const SendOptions& send_options),
              (override));

  MOCK_METHOD(std::vector<SendStatus>,
              SendMany,
              (rtc::ArrayView<DcSctpMessage> messages,
               const SendOptions& send_options),
              (override));

  MOCK_METHOD(ResetStreamsStatus,
              ResetStreams,
              (rtc::ArrayView<const StreamID> outgoing_streams),
              (override));

  MOCK_METHOD(size_t, buffered_amount, (StreamID stream_id), (const, override));

  MOCK_METHOD(size_t,
              buffered_amount_low_threshold,
              (StreamID stream_id),
              (const, override));

  MOCK_METHOD(void,
              SetBufferedAmountLowThreshold,
              (StreamID stream_id, size_t bytes),
              (override));

  MOCK_METHOD(std::optional<Metrics>, GetMetrics, (), (const, override));

  MOCK_METHOD(HandoverReadinessStatus,
              GetHandoverReadiness,
              (),
              (const, override));
  MOCK_METHOD(std::optional<DcSctpSocketHandoverState>,
              GetHandoverStateAndClose,
              (),
              (override));
};

}  // namespace dcsctp

#endif  // NET_DCSCTP_PUBLIC_MOCK_DCSCTP_SOCKET_H_
