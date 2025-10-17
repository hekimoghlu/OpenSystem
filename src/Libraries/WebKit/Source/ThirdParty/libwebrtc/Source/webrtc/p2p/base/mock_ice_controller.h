/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 7, 2024.
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
#ifndef P2P_BASE_MOCK_ICE_CONTROLLER_H_
#define P2P_BASE_MOCK_ICE_CONTROLLER_H_

#include <memory>
#include <vector>

#include "p2p/base/ice_controller_factory_interface.h"
#include "p2p/base/ice_controller_interface.h"
#include "test/gmock.h"

namespace cricket {

class MockIceController : public cricket::IceControllerInterface {
 public:
  explicit MockIceController(const cricket::IceControllerFactoryArgs& args) {}
  ~MockIceController() override = default;

  MOCK_METHOD(void, SetIceConfig, (const cricket::IceConfig&), (override));
  MOCK_METHOD(void,
              SetSelectedConnection,
              (const cricket::Connection*),
              (override));
  MOCK_METHOD(void, AddConnection, (const cricket::Connection*), (override));
  MOCK_METHOD(void,
              OnConnectionDestroyed,
              (const cricket::Connection*),
              (override));
  MOCK_METHOD(rtc::ArrayView<const cricket::Connection* const>,
              GetConnections,
              (),
              (const, override));
  MOCK_METHOD(rtc::ArrayView<const cricket::Connection*>,
              connections,
              (),
              (const, override));
  MOCK_METHOD(bool, HasPingableConnection, (), (const, override));
  MOCK_METHOD(cricket::IceControllerInterface::PingResult,
              SelectConnectionToPing,
              (int64_t),
              (override));
  MOCK_METHOD(bool,
              GetUseCandidateAttr,
              (const cricket::Connection*,
               cricket::NominationMode,
               cricket::IceMode),
              (const, override));
  MOCK_METHOD(const cricket::Connection*,
              FindNextPingableConnection,
              (),
              (override));
  MOCK_METHOD(void,
              MarkConnectionPinged,
              (const cricket::Connection*),
              (override));
  MOCK_METHOD(cricket::IceControllerInterface::SwitchResult,
              ShouldSwitchConnection,
              (cricket::IceSwitchReason, const cricket::Connection*),
              (override));
  MOCK_METHOD(cricket::IceControllerInterface::SwitchResult,
              SortAndSwitchConnection,
              (cricket::IceSwitchReason),
              (override));
  MOCK_METHOD(std::vector<const cricket::Connection*>,
              PruneConnections,
              (),
              (override));
};

class MockIceControllerFactory : public cricket::IceControllerFactoryInterface {
 public:
  ~MockIceControllerFactory() override = default;

  std::unique_ptr<cricket::IceControllerInterface> Create(
      const cricket::IceControllerFactoryArgs& args) override {
    RecordIceControllerCreated();
    return std::make_unique<MockIceController>(args);
  }

  MOCK_METHOD(void, RecordIceControllerCreated, ());
};

}  // namespace cricket

#endif  // P2P_BASE_MOCK_ICE_CONTROLLER_H_
