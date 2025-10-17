/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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
#ifndef P2P_BASE_MOCK_ICE_TRANSPORT_H_
#define P2P_BASE_MOCK_ICE_TRANSPORT_H_

#include <memory>
#include <string>
#include <vector>

#include "p2p/base/ice_transport_internal.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

using ::testing::_;
using ::testing::Return;

namespace cricket {

// Used in Chromium/remoting/protocol/channel_socket_adapter_unittest.cc
class MockIceTransport : public IceTransportInternal {
 public:
  MockIceTransport() {
    SignalReadyToSend(this);
    SignalWritableState(this);
  }

  MOCK_METHOD(int,
              SendPacket,
              (const char* data,
               size_t len,
               const rtc::PacketOptions& options,
               int flags),
              (override));
  MOCK_METHOD(int, SetOption, (rtc::Socket::Option opt, int value), (override));
  MOCK_METHOD(int, GetError, (), (override));
  MOCK_METHOD(cricket::IceRole, GetIceRole, (), (const, override));
  MOCK_METHOD(bool,
              GetStats,
              (cricket::IceTransportStats * ice_transport_stats),
              (override));

  IceTransportState GetState() const override {
    return IceTransportState::STATE_INIT;
  }
  webrtc::IceTransportState GetIceTransportState() const override {
    return webrtc::IceTransportState::kNew;
  }

  const std::string& transport_name() const override { return transport_name_; }
  int component() const override { return 0; }
  void SetIceRole(IceRole role) override {}
  // The ufrag and pwd in `ice_params` must be set
  // before candidate gathering can start.
  void SetIceParameters(const IceParameters& ice_params) override {}
  void SetRemoteIceParameters(const IceParameters& ice_params) override {}
  void SetRemoteIceMode(IceMode mode) override {}
  void SetIceConfig(const IceConfig& config) override {}
  std::optional<int> GetRttEstimate() override { return std::nullopt; }
  const Connection* selected_connection() const override { return nullptr; }
  std::optional<const CandidatePair> GetSelectedCandidatePair() const override {
    return std::nullopt;
  }
  void MaybeStartGathering() override {}
  void AddRemoteCandidate(const Candidate& candidate) override {}
  void RemoveRemoteCandidate(const Candidate& candidate) override {}
  void RemoveAllRemoteCandidates() override {}
  IceGatheringState gathering_state() const override {
    return IceGatheringState::kIceGatheringComplete;
  }

  bool receiving() const override { return true; }
  bool writable() const override { return true; }

 private:
  std::string transport_name_;
};

}  // namespace cricket

#endif  // P2P_BASE_MOCK_ICE_TRANSPORT_H_
