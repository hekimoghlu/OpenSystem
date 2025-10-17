/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 13, 2024.
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
#ifndef NET_DCSCTP_SOCKET_MOCK_CONTEXT_H_
#define NET_DCSCTP_SOCKET_MOCK_CONTEXT_H_

#include <cstdint>
#include <optional>

#include "absl/strings/string_view.h"
#include "net/dcsctp/packet/sctp_packet.h"
#include "net/dcsctp/public/dcsctp_options.h"
#include "net/dcsctp/public/dcsctp_socket.h"
#include "net/dcsctp/socket/context.h"
#include "net/dcsctp/socket/mock_dcsctp_socket_callbacks.h"
#include "test/gmock.h"

namespace dcsctp {

class MockContext : public Context {
 public:
  static constexpr TSN MyInitialTsn() { return TSN(990); }
  static constexpr TSN PeerInitialTsn() { return TSN(10); }
  static constexpr VerificationTag PeerVerificationTag() {
    return VerificationTag(0x01234567);
  }

  explicit MockContext(MockDcSctpSocketCallbacks* callbacks)
      : callbacks_(*callbacks) {
    ON_CALL(*this, is_connection_established)
        .WillByDefault(testing::Return(true));
    ON_CALL(*this, my_initial_tsn)
        .WillByDefault(testing::Return(MyInitialTsn()));
    ON_CALL(*this, peer_initial_tsn)
        .WillByDefault(testing::Return(PeerInitialTsn()));
    ON_CALL(*this, callbacks).WillByDefault(testing::ReturnRef(callbacks_));
    ON_CALL(*this, current_rto)
        .WillByDefault(testing::Return(webrtc::TimeDelta::Millis(123)));
    ON_CALL(*this, Send).WillByDefault([this](SctpPacket::Builder& builder) {
      callbacks_.SendPacketWithStatus(builder.Build());
    });
  }

  MOCK_METHOD(bool, is_connection_established, (), (const, override));
  MOCK_METHOD(TSN, my_initial_tsn, (), (const, override));
  MOCK_METHOD(TSN, peer_initial_tsn, (), (const, override));
  MOCK_METHOD(DcSctpSocketCallbacks&, callbacks, (), (const, override));

  MOCK_METHOD(void, ObserveRTT, (webrtc::TimeDelta rtt), (override));
  MOCK_METHOD(webrtc::TimeDelta, current_rto, (), (const, override));
  MOCK_METHOD(bool,
              IncrementTxErrorCounter,
              (absl::string_view reason),
              (override));
  MOCK_METHOD(void, ClearTxErrorCounter, (), (override));
  MOCK_METHOD(bool, HasTooManyTxErrors, (), (const, override));
  SctpPacket::Builder PacketBuilder() const override {
    return SctpPacket::Builder(PeerVerificationTag(), options_);
  }
  MOCK_METHOD(void, Send, (SctpPacket::Builder & builder), (override));

  DcSctpOptions options_;
  MockDcSctpSocketCallbacks& callbacks_;
};
}  // namespace dcsctp

#endif  // NET_DCSCTP_SOCKET_MOCK_CONTEXT_H_
