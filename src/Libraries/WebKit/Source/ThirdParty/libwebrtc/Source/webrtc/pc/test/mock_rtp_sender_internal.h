/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 5, 2022.
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
#ifndef PC_TEST_MOCK_RTP_SENDER_INTERNAL_H_
#define PC_TEST_MOCK_RTP_SENDER_INTERNAL_H_

#include <memory>
#include <string>
#include <vector>

#include "pc/rtp_sender.h"
#include "test/gmock.h"

namespace webrtc {

// The definition of MockRtpSender is copied in to avoid multiple inheritance.
class MockRtpSenderInternal : public RtpSenderInternal {
 public:
  // RtpSenderInterface methods.
  MOCK_METHOD(bool, SetTrack, (MediaStreamTrackInterface*), (override));
  MOCK_METHOD(rtc::scoped_refptr<MediaStreamTrackInterface>,
              track,
              (),
              (const, override));
  MOCK_METHOD(uint32_t, ssrc, (), (const, override));
  MOCK_METHOD(rtc::scoped_refptr<DtlsTransportInterface>,
              dtls_transport,
              (),
              (const, override));
  MOCK_METHOD(cricket::MediaType, media_type, (), (const, override));
  MOCK_METHOD(std::string, id, (), (const, override));
  MOCK_METHOD(std::vector<std::string>, stream_ids, (), (const, override));
  MOCK_METHOD(std::vector<RtpEncodingParameters>,
              init_send_encodings,
              (),
              (const, override));
  MOCK_METHOD(void,
              set_transport,
              (rtc::scoped_refptr<DtlsTransportInterface>),
              (override));
  MOCK_METHOD(RtpParameters, GetParameters, (), (const, override));
  MOCK_METHOD(RtpParameters, GetParametersInternal, (), (const, override));
  MOCK_METHOD(RtpParameters,
              GetParametersInternalWithAllLayers,
              (),
              (const, override));
  MOCK_METHOD(RTCError, SetParameters, (const RtpParameters&), (override));
  MOCK_METHOD(void,
              SetParametersAsync,
              (const RtpParameters&, SetParametersCallback),
              (override));
  MOCK_METHOD(void,
              SetParametersInternal,
              (const RtpParameters&, SetParametersCallback, bool blocking),
              (override));
  MOCK_METHOD(RTCError,
              SetParametersInternalWithAllLayers,
              (const RtpParameters&),
              (override));
  MOCK_METHOD(RTCError,
              CheckCodecParameters,
              (const RtpParameters&),
              (override));
  MOCK_METHOD(void, SetSendCodecs, (std::vector<cricket::Codec>), (override));
  MOCK_METHOD(std::vector<cricket::Codec>,
              GetSendCodecs,
              (),
              (const, override));
  MOCK_METHOD(rtc::scoped_refptr<DtmfSenderInterface>,
              GetDtmfSender,
              (),
              (const, override));
  MOCK_METHOD(void,
              SetFrameEncryptor,
              (rtc::scoped_refptr<FrameEncryptorInterface>),
              (override));
  MOCK_METHOD(rtc::scoped_refptr<FrameEncryptorInterface>,
              GetFrameEncryptor,
              (),
              (const, override));
  MOCK_METHOD(void,
              SetFrameTransformer,
              (rtc::scoped_refptr<FrameTransformerInterface>),
              (override));
  MOCK_METHOD(void,
              SetEncoderSelector,
              (std::unique_ptr<VideoEncoderFactory::EncoderSelectorInterface>),
              (override));
  MOCK_METHOD(void, SetObserver, (RtpSenderObserverInterface*), (override));

  // RtpSenderInternal methods.
  MOCK_METHOD1(SetMediaChannel, void(cricket::MediaSendChannelInterface*));
  MOCK_METHOD1(SetSsrc, void(uint32_t));
  MOCK_METHOD1(set_stream_ids, void(const std::vector<std::string>&));
  MOCK_METHOD1(SetStreams, void(const std::vector<std::string>&));
  MOCK_METHOD1(set_init_send_encodings,
               void(const std::vector<RtpEncodingParameters>&));
  MOCK_METHOD0(Stop, void());
  MOCK_CONST_METHOD0(AttachmentId, int());
  MOCK_METHOD1(DisableEncodingLayers,
               RTCError(const std::vector<std::string>&));
  MOCK_METHOD0(SetTransceiverAsStopped, void());
  MOCK_METHOD(void, NotifyFirstPacketSent, (), (override));
};

}  // namespace webrtc

#endif  // PC_TEST_MOCK_RTP_SENDER_INTERNAL_H_
