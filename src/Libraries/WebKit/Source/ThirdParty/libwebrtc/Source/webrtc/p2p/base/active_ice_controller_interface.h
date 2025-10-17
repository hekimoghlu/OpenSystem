/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 17, 2023.
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
#ifndef P2P_BASE_ACTIVE_ICE_CONTROLLER_INTERFACE_H_
#define P2P_BASE_ACTIVE_ICE_CONTROLLER_INTERFACE_H_

#include <optional>

#include "api/array_view.h"
#include "p2p/base/connection.h"
#include "p2p/base/ice_switch_reason.h"
#include "p2p/base/ice_transport_internal.h"
#include "p2p/base/transport_description.h"

namespace cricket {

// ActiveIceControllerInterface defines the methods for a module that actively
// manages the connection used by an ICE transport.
//
// An active ICE controller receives updates from the ICE transport when
//   - the connections state is mutated
//   - a new connection should be selected as a result of an external event (eg.
//     a different connection nominated by the remote peer)
//
// The active ICE controller takes the appropriate decisions and requests the
// ICE agent to perform the necessary actions through the IceAgentInterface.
class ActiveIceControllerInterface {
 public:
  virtual ~ActiveIceControllerInterface() = default;

  // Sets the current ICE configuration.
  virtual void SetIceConfig(const IceConfig& config) = 0;

  // Called when a new connection is added to the ICE transport.
  virtual void OnConnectionAdded(const Connection* connection) = 0;

  // Called when the transport switches that connection in active use.
  virtual void OnConnectionSwitched(const Connection* connection) = 0;

  // Called when a connection is destroyed.
  virtual void OnConnectionDestroyed(const Connection* connection) = 0;

  // Called when a STUN ping has been sent on a connection. This does not
  // indicate that a STUN response has been received.
  virtual void OnConnectionPinged(const Connection* connection) = 0;

  // Called when one of the following changes for a connection.
  // - rtt estimate
  // - write state
  // - receiving
  // - connected
  // - nominated
  virtual void OnConnectionUpdated(const Connection* connection) = 0;

  // Compute "STUN_ATTR_USE_CANDIDATE" for a STUN ping on the given connection.
  virtual bool GetUseCandidateAttribute(const Connection* connection,
                                        NominationMode mode,
                                        IceMode remote_ice_mode) const = 0;

  // Called to enque a request to pick and switch to the best available
  // connection.
  virtual void OnSortAndSwitchRequest(IceSwitchReason reason) = 0;

  // Called to pick and switch to the best available connection immediately.
  virtual void OnImmediateSortAndSwitchRequest(IceSwitchReason reason) = 0;

  // Called to switch to the given connection immediately without checking for
  // the best available connection.
  virtual bool OnImmediateSwitchRequest(IceSwitchReason reason,
                                        const Connection* selected) = 0;

  // Only for unit tests
  virtual const Connection* FindNextPingableConnection() = 0;
};

}  // namespace cricket

#endif  // P2P_BASE_ACTIVE_ICE_CONTROLLER_INTERFACE_H_
