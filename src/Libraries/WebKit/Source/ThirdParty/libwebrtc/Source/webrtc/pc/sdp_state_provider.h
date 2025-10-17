/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 29, 2022.
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
#ifndef PC_SDP_STATE_PROVIDER_H_
#define PC_SDP_STATE_PROVIDER_H_

#include <string>

#include "api/jsep.h"
#include "api/peer_connection_interface.h"

namespace webrtc {

// This interface provides access to the state of an SDP offer/answer
// negotiation.
//
// All the functions are const, so using this interface serves as
// assurance that the user is not modifying the state.
class SdpStateProvider {
 public:
  virtual ~SdpStateProvider() {}

  virtual PeerConnectionInterface::SignalingState signaling_state() const = 0;

  virtual const SessionDescriptionInterface* local_description() const = 0;
  virtual const SessionDescriptionInterface* remote_description() const = 0;
  virtual const SessionDescriptionInterface* current_local_description()
      const = 0;
  virtual const SessionDescriptionInterface* current_remote_description()
      const = 0;
  virtual const SessionDescriptionInterface* pending_local_description()
      const = 0;
  virtual const SessionDescriptionInterface* pending_remote_description()
      const = 0;

  // Whether an ICE restart has been asked for. Used in CreateOffer.
  virtual bool NeedsIceRestart(const std::string& content_name) const = 0;
  // Whether an ICE restart was indicated in the remote offer.
  // Used in CreateAnswer.
  virtual bool IceRestartPending(const std::string& content_name) const = 0;
  virtual std::optional<rtc::SSLRole> GetDtlsRole(
      const std::string& mid) const = 0;
};

}  // namespace webrtc

#endif  // PC_SDP_STATE_PROVIDER_H_
