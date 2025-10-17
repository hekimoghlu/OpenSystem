/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 13, 2025.
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
#ifndef P2P_BASE_ICE_SWITCH_REASON_H_
#define P2P_BASE_ICE_SWITCH_REASON_H_

#include <string>

#include "rtc_base/system/rtc_export.h"

namespace cricket {

enum class IceSwitchReason {
  UNKNOWN,
  REMOTE_CANDIDATE_GENERATION_CHANGE,
  NETWORK_PREFERENCE_CHANGE,
  NEW_CONNECTION_FROM_LOCAL_CANDIDATE,
  NEW_CONNECTION_FROM_REMOTE_CANDIDATE,
  NEW_CONNECTION_FROM_UNKNOWN_REMOTE_ADDRESS,
  NOMINATION_ON_CONTROLLED_SIDE,
  DATA_RECEIVED,
  CONNECT_STATE_CHANGE,
  SELECTED_CONNECTION_DESTROYED,
  // The ICE_CONTROLLER_RECHECK enum value lets an IceController request
  // P2PTransportChannel to recheck a switch periodically without an event
  // taking place.
  ICE_CONTROLLER_RECHECK,
  // The webrtc application requested a connection switch.
  APPLICATION_REQUESTED,
};

RTC_EXPORT std::string IceSwitchReasonToString(IceSwitchReason reason);

}  // namespace cricket

#endif  // P2P_BASE_ICE_SWITCH_REASON_H_
