/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 1, 2023.
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
#ifndef P2P_BASE_MOCK_ACTIVE_ICE_CONTROLLER_H_
#define P2P_BASE_MOCK_ACTIVE_ICE_CONTROLLER_H_

#include <memory>

#include "p2p/base/active_ice_controller_factory_interface.h"
#include "p2p/base/active_ice_controller_interface.h"
#include "test/gmock.h"

namespace cricket {

class MockActiveIceController : public cricket::ActiveIceControllerInterface {
 public:
  explicit MockActiveIceController(
      const cricket::ActiveIceControllerFactoryArgs& args) {}
  ~MockActiveIceController() override = default;

  MOCK_METHOD(void, SetIceConfig, (const cricket::IceConfig&), (override));
  MOCK_METHOD(void,
              OnConnectionAdded,
              (const cricket::Connection*),
              (override));
  MOCK_METHOD(void,
              OnConnectionSwitched,
              (const cricket::Connection*),
              (override));
  MOCK_METHOD(void,
              OnConnectionDestroyed,
              (const cricket::Connection*),
              (override));
  MOCK_METHOD(void,
              OnConnectionPinged,
              (const cricket::Connection*),
              (override));
  MOCK_METHOD(void,
              OnConnectionUpdated,
              (const cricket::Connection*),
              (override));
  MOCK_METHOD(bool,
              GetUseCandidateAttribute,
              (const cricket::Connection*,
               cricket::NominationMode,
               cricket::IceMode),
              (const, override));
  MOCK_METHOD(void,
              OnSortAndSwitchRequest,
              (cricket::IceSwitchReason),
              (override));
  MOCK_METHOD(void,
              OnImmediateSortAndSwitchRequest,
              (cricket::IceSwitchReason),
              (override));
  MOCK_METHOD(bool,
              OnImmediateSwitchRequest,
              (cricket::IceSwitchReason, const cricket::Connection*),
              (override));
  MOCK_METHOD(const cricket::Connection*,
              FindNextPingableConnection,
              (),
              (override));
};

class MockActiveIceControllerFactory
    : public cricket::ActiveIceControllerFactoryInterface {
 public:
  ~MockActiveIceControllerFactory() override = default;

  std::unique_ptr<cricket::ActiveIceControllerInterface> Create(
      const cricket::ActiveIceControllerFactoryArgs& args) {
    RecordActiveIceControllerCreated();
    return std::make_unique<MockActiveIceController>(args);
  }

  MOCK_METHOD(void, RecordActiveIceControllerCreated, ());
};

}  // namespace cricket

#endif  // P2P_BASE_MOCK_ACTIVE_ICE_CONTROLLER_H_
