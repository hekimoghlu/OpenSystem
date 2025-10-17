/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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
#include "p2p/client/turn_port_factory.h"

#include <memory>
#include <utility>

#include "p2p/base/port_allocator.h"
#include "p2p/base/turn_port.h"

namespace cricket {

TurnPortFactory::~TurnPortFactory() {}

std::unique_ptr<Port> TurnPortFactory::Create(
    const CreateRelayPortArgs& args,
    rtc::AsyncPacketSocket* udp_socket) {
  auto port = TurnPort::Create(args, udp_socket);
  if (!port)
    return nullptr;
  port->SetTlsCertPolicy(args.config->tls_cert_policy);
  port->SetTurnLoggingId(args.config->turn_logging_id);
  return std::move(port);
}

std::unique_ptr<Port> TurnPortFactory::Create(const CreateRelayPortArgs& args,
                                              int min_port,
                                              int max_port) {
  auto port = TurnPort::Create(args, min_port, max_port);
  if (!port)
    return nullptr;
  port->SetTlsCertPolicy(args.config->tls_cert_policy);
  port->SetTurnLoggingId(args.config->turn_logging_id);
  return std::move(port);
}

}  // namespace cricket
