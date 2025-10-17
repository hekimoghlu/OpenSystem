/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 7, 2024.
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
#ifndef P2P_CLIENT_RELAY_PORT_FACTORY_INTERFACE_H_
#define P2P_CLIENT_RELAY_PORT_FACTORY_INTERFACE_H_

#include <memory>
#include <string>

#include "p2p/base/port_interface.h"
#include "rtc_base/ref_count.h"

namespace rtc {
class AsyncPacketSocket;
class Network;
class PacketSocketFactory;
class Thread;
}  // namespace rtc

namespace webrtc {
class TurnCustomizer;
class FieldTrialsView;
}  // namespace webrtc

namespace cricket {
class Port;
struct ProtocolAddress;
struct RelayServerConfig;

// A struct containing arguments to RelayPortFactory::Create()
struct CreateRelayPortArgs {
  rtc::Thread* network_thread;
  rtc::PacketSocketFactory* socket_factory;
  const rtc::Network* network;
  const ProtocolAddress* server_address;
  const RelayServerConfig* config;
  std::string username;
  std::string password;
  webrtc::TurnCustomizer* turn_customizer = nullptr;
  const webrtc::FieldTrialsView* field_trials = nullptr;
  // Relative priority of candidates from this TURN server in relation
  // to the candidates from other servers. Required because ICE priorities
  // need to be unique.
  int relative_priority = 0;
};

// A factory for creating RelayPort's.
class RelayPortFactoryInterface {
 public:
  virtual ~RelayPortFactoryInterface() {}

  // This variant is used for UDP connection to the relay server
  // using a already existing shared socket.
  virtual std::unique_ptr<Port> Create(const CreateRelayPortArgs& args,
                                       rtc::AsyncPacketSocket* udp_socket) = 0;

  // This variant is used for the other cases.
  virtual std::unique_ptr<Port> Create(const CreateRelayPortArgs& args,
                                       int min_port,
                                       int max_port) = 0;
};

}  // namespace cricket

#endif  // P2P_CLIENT_RELAY_PORT_FACTORY_INTERFACE_H_
