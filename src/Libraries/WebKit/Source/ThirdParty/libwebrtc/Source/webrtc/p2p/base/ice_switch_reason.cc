/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 26, 2023.
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
#include "p2p/base/ice_switch_reason.h"

#include <string>

namespace cricket {

std::string IceSwitchReasonToString(IceSwitchReason reason) {
  switch (reason) {
    case IceSwitchReason::REMOTE_CANDIDATE_GENERATION_CHANGE:
      return "remote candidate generation maybe changed";
    case IceSwitchReason::NETWORK_PREFERENCE_CHANGE:
      return "network preference changed";
    case IceSwitchReason::NEW_CONNECTION_FROM_LOCAL_CANDIDATE:
      return "new candidate pairs created from a new local candidate";
    case IceSwitchReason::NEW_CONNECTION_FROM_REMOTE_CANDIDATE:
      return "new candidate pairs created from a new remote candidate";
    case IceSwitchReason::NEW_CONNECTION_FROM_UNKNOWN_REMOTE_ADDRESS:
      return "a new candidate pair created from an unknown remote address";
    case IceSwitchReason::NOMINATION_ON_CONTROLLED_SIDE:
      return "nomination on the controlled side";
    case IceSwitchReason::DATA_RECEIVED:
      return "data received";
    case IceSwitchReason::CONNECT_STATE_CHANGE:
      return "candidate pair state changed";
    case IceSwitchReason::SELECTED_CONNECTION_DESTROYED:
      return "selected candidate pair destroyed";
    case IceSwitchReason::ICE_CONTROLLER_RECHECK:
      return "ice-controller-request-recheck";
    case IceSwitchReason::APPLICATION_REQUESTED:
      return "application requested";
    case IceSwitchReason::UNKNOWN:
    default:
      return "unknown";
  }
}

}  // namespace cricket
