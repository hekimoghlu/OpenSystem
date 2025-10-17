/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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
#ifndef P2P_BASE_MOCK_ICE_AGENT_H_
#define P2P_BASE_MOCK_ICE_AGENT_H_

#include <vector>

#include "p2p/base/connection.h"
#include "p2p/base/ice_agent_interface.h"
#include "p2p/base/ice_switch_reason.h"
#include "p2p/base/transport_description.h"
#include "test/gmock.h"

namespace cricket {

class MockIceAgent : public IceAgentInterface {
 public:
  ~MockIceAgent() override = default;

  MOCK_METHOD(int64_t, GetLastPingSentMs, (), (override, const));
  MOCK_METHOD(IceRole, GetIceRole, (), (override, const));
  MOCK_METHOD(void, OnStartedPinging, (), (override));
  MOCK_METHOD(void, UpdateConnectionStates, (), (override));
  MOCK_METHOD(void, UpdateState, (), (override));
  MOCK_METHOD(void,
              ForgetLearnedStateForConnections,
              (rtc::ArrayView<const Connection* const>),
              (override));
  MOCK_METHOD(void, SendPingRequest, (const Connection*), (override));
  MOCK_METHOD(void,
              SwitchSelectedConnection,
              (const Connection*, IceSwitchReason),
              (override));
  MOCK_METHOD(bool,
              PruneConnections,
              (rtc::ArrayView<const Connection* const>),
              (override));
};

}  // namespace cricket

#endif  // P2P_BASE_MOCK_ICE_AGENT_H_
