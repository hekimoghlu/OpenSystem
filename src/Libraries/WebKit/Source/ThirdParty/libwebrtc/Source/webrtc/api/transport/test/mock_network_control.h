/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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
#ifndef API_TRANSPORT_TEST_MOCK_NETWORK_CONTROL_H_
#define API_TRANSPORT_TEST_MOCK_NETWORK_CONTROL_H_

#include "api/transport/network_control.h"
#include "test/gmock.h"

namespace webrtc {

class MockNetworkControllerInterface : public NetworkControllerInterface {
 public:
  MOCK_METHOD(NetworkControlUpdate,
              OnNetworkAvailability,
              (NetworkAvailability),
              (override));
  MOCK_METHOD(NetworkControlUpdate,
              OnProcessInterval,
              (ProcessInterval),
              (override));
  MOCK_METHOD(NetworkControlUpdate,
              OnNetworkRouteChange,
              (NetworkRouteChange),
              (override));
  MOCK_METHOD(NetworkControlUpdate,
              OnRemoteBitrateReport,
              (RemoteBitrateReport),
              (override));
  MOCK_METHOD(NetworkControlUpdate,
              OnRoundTripTimeUpdate,
              (RoundTripTimeUpdate),
              (override));
  MOCK_METHOD(NetworkControlUpdate, OnSentPacket, (SentPacket), (override));
  MOCK_METHOD(NetworkControlUpdate,
              OnReceivedPacket,
              (ReceivedPacket),
              (override));
  MOCK_METHOD(NetworkControlUpdate,
              OnStreamsConfig,
              (StreamsConfig),
              (override));
  MOCK_METHOD(NetworkControlUpdate,
              OnTargetRateConstraints,
              (TargetRateConstraints),
              (override));
  MOCK_METHOD(NetworkControlUpdate,
              OnTransportLossReport,
              (TransportLossReport),
              (override));
  MOCK_METHOD(NetworkControlUpdate,
              OnTransportPacketsFeedback,
              (TransportPacketsFeedback),
              (override));
  MOCK_METHOD(NetworkControlUpdate,
              OnNetworkStateEstimate,
              (NetworkStateEstimate),
              (override));
};

class MockNetworkStateEstimator : public NetworkStateEstimator {
 public:
  MOCK_METHOD(std::optional<NetworkStateEstimate>,
              GetCurrentEstimate,
              (),
              (override));
  MOCK_METHOD(void,
              OnTransportPacketsFeedback,
              (const TransportPacketsFeedback&),
              (override));
  MOCK_METHOD(void, OnReceivedPacket, (const PacketResult&), (override));
  MOCK_METHOD(void, OnRouteChange, (const NetworkRouteChange&), (override));
};

}  // namespace webrtc

#endif  // API_TRANSPORT_TEST_MOCK_NETWORK_CONTROL_H_
