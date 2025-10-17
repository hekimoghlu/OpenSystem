/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 29, 2024.
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
#include "rtc_base/openssl_adapter.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rtc_base/socket.h"
#include "rtc_base/socket_address.h"
#include "rtc_base/ssl_certificate.h"
#include "rtc_base/thread.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace rtc {
namespace {

class MockAsyncSocket : public Socket {
 public:
  virtual ~MockAsyncSocket() = default;
  MOCK_METHOD(Socket*, Accept, (SocketAddress*), (override));
  MOCK_METHOD(SocketAddress, GetLocalAddress, (), (const, override));
  MOCK_METHOD(SocketAddress, GetRemoteAddress, (), (const, override));
  MOCK_METHOD(int, Bind, (const SocketAddress&), (override));
  MOCK_METHOD(int, Connect, (const SocketAddress&), (override));
  MOCK_METHOD(int, Send, (const void*, size_t), (override));
  MOCK_METHOD(int,
              SendTo,
              (const void*, size_t, const SocketAddress&),
              (override));
  MOCK_METHOD(int, Recv, (void*, size_t, int64_t*), (override));
  MOCK_METHOD(int,
              RecvFrom,
              (void*, size_t, SocketAddress*, int64_t*),
              (override));
  MOCK_METHOD(int, Listen, (int), (override));
  MOCK_METHOD(int, Close, (), (override));
  MOCK_METHOD(int, GetError, (), (const, override));
  MOCK_METHOD(void, SetError, (int), (override));
  MOCK_METHOD(ConnState, GetState, (), (const, override));
  MOCK_METHOD(int, GetOption, (Option, int*), (override));
  MOCK_METHOD(int, SetOption, (Option, int), (override));
};

class MockCertVerifier : public SSLCertificateVerifier {
 public:
  virtual ~MockCertVerifier() = default;
  MOCK_METHOD(bool, Verify, (const SSLCertificate&), (override));
};

}  // namespace

using ::testing::_;
using ::testing::Return;

TEST(OpenSSLAdapterTest, TestTransformAlpnProtocols) {
  EXPECT_EQ("", TransformAlpnProtocols(std::vector<std::string>()));

  // Protocols larger than 255 characters (whose size can't be fit in a byte),
  // can't be converted, and an empty string will be returned.
  std::string large_protocol(256, 'a');
  EXPECT_EQ("",
            TransformAlpnProtocols(std::vector<std::string>{large_protocol}));

  // One protocol test.
  std::vector<std::string> alpn_protos{"h2"};
  std::stringstream expected_response;
  expected_response << static_cast<char>(2) << "h2";
  EXPECT_EQ(expected_response.str(), TransformAlpnProtocols(alpn_protos));

  // Standard protocols test (h2,http/1.1).
  alpn_protos.push_back("http/1.1");
  expected_response << static_cast<char>(8) << "http/1.1";
  EXPECT_EQ(expected_response.str(), TransformAlpnProtocols(alpn_protos));
}

// Verifies that SSLStart works when OpenSSLAdapter is started in standalone
// mode.
TEST(OpenSSLAdapterTest, TestBeginSSLBeforeConnection) {
  rtc::AutoThread main_thread;
  Socket* async_socket = new MockAsyncSocket();
  OpenSSLAdapter adapter(async_socket);
  EXPECT_EQ(adapter.StartSSL("webrtc.org"), 0);
}

// Verifies that the adapter factory can create new adapters.
TEST(OpenSSLAdapterFactoryTest, CreateSingleOpenSSLAdapter) {
  rtc::AutoThread main_thread;
  OpenSSLAdapterFactory adapter_factory;
  Socket* async_socket = new MockAsyncSocket();
  auto simple_adapter = std::unique_ptr<OpenSSLAdapter>(
      adapter_factory.CreateAdapter(async_socket));
  EXPECT_NE(simple_adapter, nullptr);
}

// Verifies that setting a custom verifier still allows for adapters to be
// created.
TEST(OpenSSLAdapterFactoryTest, CreateWorksWithCustomVerifier) {
  rtc::AutoThread main_thread;
  MockCertVerifier* mock_verifier = new MockCertVerifier();
  EXPECT_CALL(*mock_verifier, Verify(_)).WillRepeatedly(Return(true));
  auto cert_verifier = std::unique_ptr<SSLCertificateVerifier>(mock_verifier);

  OpenSSLAdapterFactory adapter_factory;
  adapter_factory.SetCertVerifier(cert_verifier.get());
  Socket* async_socket = new MockAsyncSocket();
  auto simple_adapter = std::unique_ptr<OpenSSLAdapter>(
      adapter_factory.CreateAdapter(async_socket));
  EXPECT_NE(simple_adapter, nullptr);
}

}  // namespace rtc
