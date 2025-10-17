/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 28, 2023.
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
#ifndef P2P_CLIENT_TURN_PORT_FACTORY_H_
#define P2P_CLIENT_TURN_PORT_FACTORY_H_

#include <memory>

#include "p2p/base/port.h"
#include "p2p/client/relay_port_factory_interface.h"
#include "rtc_base/async_packet_socket.h"

namespace cricket {

// This is a RelayPortFactory that produces TurnPorts.
class TurnPortFactory : public RelayPortFactoryInterface {
 public:
  ~TurnPortFactory() override;

  std::unique_ptr<Port> Create(const CreateRelayPortArgs& args,
                               rtc::AsyncPacketSocket* udp_socket) override;

  std::unique_ptr<Port> Create(const CreateRelayPortArgs& args,
                               int min_port,
                               int max_port) override;
};

}  // namespace cricket

#endif  // P2P_CLIENT_TURN_PORT_FACTORY_H_
