/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 15, 2025.
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
#ifndef P2P_BASE_ICE_AGENT_INTERFACE_H_
#define P2P_BASE_ICE_AGENT_INTERFACE_H_

#include "api/array_view.h"
#include "p2p/base/connection.h"
#include "p2p/base/ice_switch_reason.h"

namespace cricket {

// IceAgentInterface provides methods that allow an ICE controller to manipulate
// the connections available to a transport, and used by the transport to
// transfer data.
class IceAgentInterface {
 public:
  virtual ~IceAgentInterface() = default;

  // Get the time when the last ping was sent.
  // This is only needed in some scenarios if the agent decides to ping on its
  // own, eg. in some switchover scenarios. Otherwise the ICE controller could
  // keep this state on its own.
  // TODO(bugs.webrtc.org/14367): route extra pings through the ICE controller.
  virtual int64_t GetLastPingSentMs() const = 0;

  // Get the ICE role of this ICE agent.
  virtual IceRole GetIceRole() const = 0;

  // Called when a pingable connection first becomes available.
  virtual void OnStartedPinging() = 0;

  // Update the state of all available connections.
  virtual void UpdateConnectionStates() = 0;

  // Update the internal state of the ICE agent. An ICE controller should call
  // this at the end of a sequence of actions to combine several mutations into
  // a single state refresh.
  // TODO(bugs.webrtc.org/14431): ICE agent state updates should be internal to
  // the agent. If batching is necessary, use a more appropriate interface.
  virtual void UpdateState() = 0;

  // Reset the given connections to a state of newly connected connections.
  // - STATE_WRITE_INIT
  // - receving = false
  // - throw away all pending request
  // - reset RttEstimate
  //
  // Keep the following unchanged:
  // - connected
  // - remote_candidate
  // - statistics
  //
  // SignalStateChange will not be triggered.
  virtual void ForgetLearnedStateForConnections(
      rtc::ArrayView<const Connection* const> connections) = 0;

  // Send a STUN ping request for the given connection.
  virtual void SendPingRequest(const Connection* connection) = 0;

  // Switch the transport to use the given connection.
  virtual void SwitchSelectedConnection(const Connection* new_connection,
                                        IceSwitchReason reason) = 0;

  // Prune away the given connections. Returns true if pruning is permitted and
  // successfully performed.
  virtual bool PruneConnections(
      rtc::ArrayView<const Connection* const> connections) = 0;
};

}  // namespace cricket

#endif  // P2P_BASE_ICE_AGENT_INTERFACE_H_
