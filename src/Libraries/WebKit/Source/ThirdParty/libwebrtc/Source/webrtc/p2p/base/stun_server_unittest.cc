/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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
#include "p2p/base/stun_server.h"

#include <string.h>

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "rtc_base/byte_buffer.h"
#include "rtc_base/ip_address.h"
#include "rtc_base/logging.h"
#include "rtc_base/test_client.h"
#include "rtc_base/thread.h"
#include "rtc_base/virtual_socket_server.h"
#include "test/gtest.h"

namespace cricket {

namespace {
const rtc::SocketAddress server_addr("99.99.99.1", 3478);
const rtc::SocketAddress client_addr("1.2.3.4", 1234);
}  // namespace

class StunServerTest : public ::testing::Test {
 public:
  StunServerTest() : ss_(new rtc::VirtualSocketServer()) {
    ss_->SetMessageQueue(&main_thread);
    server_.reset(
        new StunServer(rtc::AsyncUDPSocket::Create(ss_.get(), server_addr)));
    client_.reset(new rtc::TestClient(
        absl::WrapUnique(rtc::AsyncUDPSocket::Create(ss_.get(), client_addr))));
  }

  void Send(const StunMessage& msg) {
    rtc::ByteBufferWriter buf;
    msg.Write(&buf);
    Send(reinterpret_cast<const char*>(buf.Data()),
         static_cast<int>(buf.Length()));
  }
  void Send(const char* buf, int len) {
    client_->SendTo(buf, len, server_addr);
  }
  bool ReceiveFails() { return (client_->CheckNoPacket()); }
  StunMessage* Receive() {
    StunMessage* msg = NULL;
    std::unique_ptr<rtc::TestClient::Packet> packet =
        client_->NextPacket(rtc::TestClient::kTimeoutMs);
    if (packet) {
      rtc::ByteBufferReader buf(packet->buf);
      msg = new StunMessage();
      msg->Read(&buf);
    }
    return msg;
  }

 private:
  rtc::AutoThread main_thread;
  std::unique_ptr<rtc::VirtualSocketServer> ss_;
  std::unique_ptr<StunServer> server_;
  std::unique_ptr<rtc::TestClient> client_;
};

TEST_F(StunServerTest, TestGood) {
  // kStunLegacyTransactionIdLength = 16 for legacy RFC 3489 request
  std::string transaction_id = "0123456789abcdef";
  StunMessage req(STUN_BINDING_REQUEST, transaction_id);
  Send(req);

  StunMessage* msg = Receive();
  ASSERT_TRUE(msg != NULL);
  EXPECT_EQ(STUN_BINDING_RESPONSE, msg->type());
  EXPECT_EQ(req.transaction_id(), msg->transaction_id());

  const StunAddressAttribute* mapped_addr =
      msg->GetAddress(STUN_ATTR_MAPPED_ADDRESS);
  EXPECT_TRUE(mapped_addr != NULL);
  EXPECT_EQ(1, mapped_addr->family());
  EXPECT_EQ(client_addr.port(), mapped_addr->port());

  delete msg;
}

TEST_F(StunServerTest, TestGoodXorMappedAddr) {
  // kStunTransactionIdLength = 12 for RFC 5389 request
  // StunMessage::Write will automatically insert magic cookie (0x2112A442)
  std::string transaction_id = "0123456789ab";
  StunMessage req(STUN_BINDING_REQUEST, transaction_id);
  Send(req);

  StunMessage* msg = Receive();
  ASSERT_TRUE(msg != NULL);
  EXPECT_EQ(STUN_BINDING_RESPONSE, msg->type());
  EXPECT_EQ(req.transaction_id(), msg->transaction_id());

  const StunAddressAttribute* mapped_addr =
      msg->GetAddress(STUN_ATTR_XOR_MAPPED_ADDRESS);
  EXPECT_TRUE(mapped_addr != NULL);
  EXPECT_EQ(1, mapped_addr->family());
  EXPECT_EQ(client_addr.port(), mapped_addr->port());

  delete msg;
}

// Send legacy RFC 3489 request, should not get xor mapped addr
TEST_F(StunServerTest, TestNoXorMappedAddr) {
  // kStunLegacyTransactionIdLength = 16 for legacy RFC 3489 request
  std::string transaction_id = "0123456789abcdef";
  StunMessage req(STUN_BINDING_REQUEST, transaction_id);
  Send(req);

  StunMessage* msg = Receive();
  ASSERT_TRUE(msg != NULL);
  EXPECT_EQ(STUN_BINDING_RESPONSE, msg->type());
  EXPECT_EQ(req.transaction_id(), msg->transaction_id());

  const StunAddressAttribute* mapped_addr =
      msg->GetAddress(STUN_ATTR_XOR_MAPPED_ADDRESS);
  EXPECT_TRUE(mapped_addr == NULL);

  delete msg;
}

TEST_F(StunServerTest, TestBad) {
  const char* bad =
      "this is a completely nonsensical message whose only "
      "purpose is to make the parser go 'ack'.  it doesn't "
      "look anything like a normal stun message";
  Send(bad, static_cast<int>(strlen(bad)));

  ASSERT_TRUE(ReceiveFails());
}

}  // namespace cricket
