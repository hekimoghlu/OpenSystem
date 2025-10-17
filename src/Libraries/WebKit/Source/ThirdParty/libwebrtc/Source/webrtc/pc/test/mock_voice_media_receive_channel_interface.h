/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 21, 2024.
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
#ifndef PC_TEST_MOCK_VOICE_MEDIA_RECEIVE_CHANNEL_INTERFACE_H_
#define PC_TEST_MOCK_VOICE_MEDIA_RECEIVE_CHANNEL_INTERFACE_H_

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "api/call/audio_sink.h"
#include "media/base/media_channel.h"
#include "media/base/media_channel_impl.h"
#include "modules/rtp_rtcp/source/rtp_packet_received.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace cricket {

class MockVoiceMediaReceiveChannelInterface
    : public VoiceMediaReceiveChannelInterface {
 public:
  MockVoiceMediaReceiveChannelInterface() {
    ON_CALL(*this, AsVoiceReceiveChannel).WillByDefault(testing::Return(this));
  }

  // VoiceMediaReceiveChannelInterface
  MOCK_METHOD(bool,
              SetReceiverParameters,
              (const AudioReceiverParameters& params),
              (override));
  MOCK_METHOD(webrtc::RtpParameters,
              GetRtpReceiverParameters,
              (uint32_t ssrc),
              (const, override));
  MOCK_METHOD(std::vector<webrtc::RtpSource>,
              GetSources,
              (uint32_t ssrc),
              (const, override));
  MOCK_METHOD(webrtc::RtpParameters,
              GetDefaultRtpReceiveParameters,
              (),
              (const, override));
  MOCK_METHOD(void, SetPlayout, (bool playout), (override));
  MOCK_METHOD(bool,
              SetOutputVolume,
              (uint32_t ssrc, double volume),
              (override));
  MOCK_METHOD(bool, SetDefaultOutputVolume, (double volume), (override));
  MOCK_METHOD(void,
              SetRawAudioSink,
              (uint32_t ssrc, std::unique_ptr<webrtc::AudioSinkInterface> sink),
              (override));
  MOCK_METHOD(void,
              SetDefaultRawAudioSink,
              (std::unique_ptr<webrtc::AudioSinkInterface> sink),
              (override));
  MOCK_METHOD(bool,
              GetStats,
              (VoiceMediaReceiveInfo * stats, bool reset_legacy),
              (override));
  MOCK_METHOD(void, SetReceiveNackEnabled, (bool enabled), (override));
  MOCK_METHOD(void, SetRtcpMode, (webrtc::RtcpMode mode), (override));
  MOCK_METHOD(void, SetReceiveNonSenderRttEnabled, (bool enabled), (override));

  // MediaReceiveChannelInterface
  MOCK_METHOD(VideoMediaReceiveChannelInterface*,
              AsVideoReceiveChannel,
              (),
              (override));
  MOCK_METHOD(VoiceMediaReceiveChannelInterface*,
              AsVoiceReceiveChannel,
              (),
              (override));
  MOCK_METHOD(cricket::MediaType, media_type, (), (const, override));
  MOCK_METHOD(bool, AddRecvStream, (const StreamParams& sp), (override));
  MOCK_METHOD(bool, RemoveRecvStream, (uint32_t ssrc), (override));
  MOCK_METHOD(void, ResetUnsignaledRecvStream, (), (override));
  MOCK_METHOD(void,
              SetInterface,
              (MediaChannelNetworkInterface * iface),
              (override));
  MOCK_METHOD(void,
              OnPacketReceived,
              (const webrtc::RtpPacketReceived& packet),
              (override));
  MOCK_METHOD(std::optional<uint32_t>,
              GetUnsignaledSsrc,
              (),
              (const, override));
  MOCK_METHOD(void,
              ChooseReceiverReportSsrc,
              (const std::set<uint32_t>& choices),
              (override));
  MOCK_METHOD(void, OnDemuxerCriteriaUpdatePending, (), (override));
  MOCK_METHOD(void, OnDemuxerCriteriaUpdateComplete, (), (override));
  MOCK_METHOD(
      void,
      SetFrameDecryptor,
      (uint32_t ssrc,
       rtc::scoped_refptr<webrtc::FrameDecryptorInterface> frame_decryptor),
      (override));
  MOCK_METHOD(
      void,
      SetDepacketizerToDecoderFrameTransformer,
      (uint32_t ssrc,
       rtc::scoped_refptr<webrtc::FrameTransformerInterface> frame_transformer),
      (override));
  MOCK_METHOD(bool,
              SetBaseMinimumPlayoutDelayMs,
              (uint32_t ssrc, int delay_ms),
              (override));
  MOCK_METHOD(std::optional<int>,
              GetBaseMinimumPlayoutDelayMs,
              (uint32_t ssrc),
              (const, override));
};

static_assert(!std::is_abstract_v<MockVoiceMediaReceiveChannelInterface>, "");

}  // namespace cricket

#endif  // PC_TEST_MOCK_VOICE_MEDIA_RECEIVE_CHANNEL_INTERFACE_H_
