/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 6, 2023.
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
#ifndef TEST_PEER_SCENARIO_SIGNALING_ROUTE_H_
#define TEST_PEER_SCENARIO_SIGNALING_ROUTE_H_

#include <string>
#include <utility>

#include "test/network/network_emulation_manager.h"
#include "test/peer_scenario/peer_scenario_client.h"

namespace webrtc {
namespace test {

// Helper class to reduce the amount of boilerplate required for ICE signalling
// ad SDP negotiation.
class SignalingRoute {
 public:
  SignalingRoute(PeerScenarioClient* caller,
                 PeerScenarioClient* callee,
                 CrossTrafficRoute* send_route,
                 CrossTrafficRoute* ret_route);

  void StartIceSignaling();

  // The `modify_offer` callback is used to modify an offer after the local
  // description has been set. This is legal (but odd) behavior.
  // The `munge_offer` callback is used to modify an offer between its creation
  // and set local description. This behavior is forbidden according to the spec
  // but available here in order to allow test coverage on corner cases.
  // `callee_remote_description_set` is invoked when callee has applied the
  // offer but not yet created an answer. The purpose is to allow tests to
  // modify transceivers created from the offer.  The `exchange_finished`
  // callback is called with the answer produced after SDP negotations has
  // completed.
  // TODO(srte): Handle lossy links.
  void NegotiateSdp(
      std::function<void(SessionDescriptionInterface* offer)> munge_offer,
      std::function<void(SessionDescriptionInterface* offer)> modify_offer,
      std::function<void()> callee_remote_description_set,
      std::function<void(const SessionDescriptionInterface& answer)>
          exchange_finished);
  void NegotiateSdp(
      std::function<void(SessionDescriptionInterface* offer)> munge_offer,
      std::function<void(SessionDescriptionInterface* offer)> modify_offer,
      std::function<void(const SessionDescriptionInterface& answer)>
          exchange_finished);
  void NegotiateSdp(
      std::function<void(SessionDescriptionInterface* offer)> modify_offer,
      std::function<void(const SessionDescriptionInterface& answer)>
          exchange_finished);
  void NegotiateSdp(
      std::function<void()> remote_description_set,
      std::function<void(const SessionDescriptionInterface& answer)>
          exchange_finished);
  void NegotiateSdp(
      std::function<void(const SessionDescriptionInterface& answer)>
          exchange_finished);
  SignalingRoute reverse() {
    return SignalingRoute(callee_, caller_, ret_route_, send_route_);
  }

 private:
  PeerScenarioClient* const caller_;
  PeerScenarioClient* const callee_;
  CrossTrafficRoute* const send_route_;
  CrossTrafficRoute* const ret_route_;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_PEER_SCENARIO_SIGNALING_ROUTE_H_
