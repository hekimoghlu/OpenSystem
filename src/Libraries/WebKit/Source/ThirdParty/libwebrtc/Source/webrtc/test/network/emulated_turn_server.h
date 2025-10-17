/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 11, 2025.
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
#ifndef TEST_NETWORK_EMULATED_TURN_SERVER_H_
#define TEST_NETWORK_EMULATED_TURN_SERVER_H_

#include <map>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "api/test/network_emulation_manager.h"
#include "api/transport/stun.h"
#include "p2p/base/turn_server.h"
#include "rtc_base/async_packet_socket.h"

namespace webrtc {
namespace test {

// EmulatedTURNServer wraps cricket::TurnServer to be used inside
// a emulated network.
//
// Packets from EmulatedEndpoint (client or peer) are received in
// EmulatedTURNServer::OnPacketReceived which performs a map lookup
// and delivers them into cricket::TurnServer using
// AsyncPacketSocket::SignalReadPacket
//
// Packets from cricket::TurnServer to EmulatedEndpoint are sent into
// using a wrapper around AsyncPacketSocket (no lookup required as the
// wrapper around AsyncPacketSocket keep a pointer to the EmulatedEndpoint).
class EmulatedTURNServer : public EmulatedTURNServerInterface,
                           public cricket::TurnAuthInterface,
                           public webrtc::EmulatedNetworkReceiverInterface {
 public:
  // Create an EmulatedTURNServer.
  // `thread` is a thread that will be used to run cricket::TurnServer
  // that expects all calls to be made from a single thread.
  EmulatedTURNServer(std::unique_ptr<rtc::Thread> thread,
                     EmulatedEndpoint* client,
                     EmulatedEndpoint* peer);
  ~EmulatedTURNServer() override;

  IceServerConfig GetIceServerConfig() const override { return ice_config_; }

  EmulatedEndpoint* GetClientEndpoint() const override { return client_; }

  rtc::SocketAddress GetClientEndpointAddress() const override {
    return client_address_;
  }

  EmulatedEndpoint* GetPeerEndpoint() const override { return peer_; }

  // cricket::TurnAuthInterface
  bool GetKey(absl::string_view username,
              absl::string_view realm,
              std::string* key) override {
    return cricket::ComputeStunCredentialHash(
        std::string(username), std::string(realm), std::string(username), key);
  }

  rtc::AsyncPacketSocket* CreatePeerSocket() { return Wrap(peer_); }

  // This method is called by network emulation when a packet
  // comes from an emulated link.
  void OnPacketReceived(webrtc::EmulatedIpPacket packet) override;

  // This is called when the TURN server deletes a socket.
  void Unbind(rtc::SocketAddress address);

  // Unbind all sockets.
  void Stop();

 private:
  std::unique_ptr<rtc::Thread> thread_;
  rtc::SocketAddress client_address_;
  IceServerConfig ice_config_;
  EmulatedEndpoint* const client_;
  EmulatedEndpoint* const peer_;
  std::unique_ptr<cricket::TurnServer> turn_server_ RTC_GUARDED_BY(&thread_);
  class AsyncPacketSocketWrapper;
  std::map<rtc::SocketAddress, AsyncPacketSocketWrapper*> sockets_
      RTC_GUARDED_BY(&thread_);

  // Wraps a EmulatedEndpoint in a AsyncPacketSocket to bridge interaction
  // with TurnServer. cricket::TurnServer gets ownership of the socket.
  rtc::AsyncPacketSocket* Wrap(EmulatedEndpoint* endpoint);
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_NETWORK_EMULATED_TURN_SERVER_H_
