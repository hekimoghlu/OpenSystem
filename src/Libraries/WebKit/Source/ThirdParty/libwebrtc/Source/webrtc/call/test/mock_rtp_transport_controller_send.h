/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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
#ifndef CALL_TEST_MOCK_RTP_TRANSPORT_CONTROLLER_SEND_H_
#define CALL_TEST_MOCK_RTP_TRANSPORT_CONTROLLER_SEND_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/crypto/crypto_options.h"
#include "api/crypto/frame_encryptor_interface.h"
#include "api/frame_transformer_interface.h"
#include "api/transport/bitrate_settings.h"
#include "call/rtp_transport_controller_send_interface.h"
#include "modules/pacing/packet_router.h"
#include "modules/rtp_rtcp/source/rtp_rtcp_interface.h"
#include "rtc_base/network/sent_packet.h"
#include "rtc_base/network_route.h"
#include "rtc_base/rate_limiter.h"
#include "test/gmock.h"

namespace webrtc {

class MockRtpTransportControllerSend
    : public RtpTransportControllerSendInterface {
 public:
  MOCK_METHOD(RtpVideoSenderInterface*,
              CreateRtpVideoSender,
              ((const std::map<uint32_t, RtpState>&),
               (const std::map<uint32_t, RtpPayloadState>&),
               const RtpConfig&,
               int rtcp_report_interval_ms,
               Transport*,
               const RtpSenderObservers&,
               std::unique_ptr<FecController>,
               const RtpSenderFrameEncryptionConfig&,
               rtc::scoped_refptr<FrameTransformerInterface>),
              (override));
  MOCK_METHOD(void,
              DestroyRtpVideoSender,
              (RtpVideoSenderInterface*),
              (override));
  MOCK_METHOD(void, RegisterSendingRtpStream, (RtpRtcpInterface&), (override));
  MOCK_METHOD(void,
              DeRegisterSendingRtpStream,
              (RtpRtcpInterface&),
              (override));
  MOCK_METHOD(PacketRouter*, packet_router, (), (override));
  MOCK_METHOD(NetworkStateEstimateObserver*,
              network_state_estimate_observer,
              (),
              (override));
  MOCK_METHOD(RtpPacketSender*, packet_sender, (), (override));
  MOCK_METHOD(void,
              SetAllocatedSendBitrateLimits,
              (BitrateAllocationLimits),
              (override));
  MOCK_METHOD(void,
              ReconfigureBandwidthEstimation,
              (const BandwidthEstimationSettings&),
              (override));
  MOCK_METHOD(void, SetPacingFactor, (float), (override));
  MOCK_METHOD(void, SetQueueTimeLimit, (int), (override));
  MOCK_METHOD(StreamFeedbackProvider*,
              GetStreamFeedbackProvider,
              (),
              (override));
  MOCK_METHOD(void,
              RegisterTargetTransferRateObserver,
              (TargetTransferRateObserver*),
              (override));
  MOCK_METHOD(void,
              OnNetworkRouteChanged,
              (absl::string_view, const rtc::NetworkRoute&),
              (override));
  MOCK_METHOD(void, OnNetworkAvailability, (bool), (override));
  MOCK_METHOD(NetworkLinkRtcpObserver*, GetRtcpObserver, (), (override));
  MOCK_METHOD(int64_t, GetPacerQueuingDelayMs, (), (const, override));
  MOCK_METHOD(std::optional<Timestamp>,
              GetFirstPacketTime,
              (),
              (const, override));
  MOCK_METHOD(void, EnablePeriodicAlrProbing, (bool), (override));
  MOCK_METHOD(void, OnSentPacket, (const rtc::SentPacket&), (override));
  MOCK_METHOD(void,
              SetSdpBitrateParameters,
              (const BitrateConstraints&),
              (override));
  MOCK_METHOD(void,
              SetClientBitratePreferences,
              (const BitrateSettings&),
              (override));
  MOCK_METHOD(void, OnTransportOverheadChanged, (size_t), (override));
  MOCK_METHOD(void, AccountForAudioPacketsInPacedSender, (bool), (override));
  MOCK_METHOD(void, IncludeOverheadInPacedSender, (), (override));
  MOCK_METHOD(void, OnReceivedPacket, (const ReceivedPacket&), (override));
  MOCK_METHOD(void, EnsureStarted, (), (override));
  MOCK_METHOD(NetworkControllerInterface*,
              GetNetworkController,
              (),
              (override));
};
}  // namespace webrtc
#endif  // CALL_TEST_MOCK_RTP_TRANSPORT_CONTROLLER_SEND_H_
