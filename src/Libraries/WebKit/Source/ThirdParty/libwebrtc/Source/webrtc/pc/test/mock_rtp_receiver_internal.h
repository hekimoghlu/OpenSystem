/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 11, 2022.
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
#ifndef PC_TEST_MOCK_RTP_RECEIVER_INTERNAL_H_
#define PC_TEST_MOCK_RTP_RECEIVER_INTERNAL_H_

#include <optional>
#include <string>
#include <vector>

#include "pc/rtp_receiver.h"
#include "test/gmock.h"

namespace webrtc {

// The definition of MockRtpReceiver is copied in to avoid multiple inheritance.
class MockRtpReceiverInternal : public RtpReceiverInternal {
 public:
  // RtpReceiverInterface methods.
  MOCK_METHOD(rtc::scoped_refptr<MediaStreamTrackInterface>,
              track,
              (),
              (const, override));
  MOCK_METHOD(rtc::scoped_refptr<DtlsTransportInterface>,
              dtls_transport,
              (),
              (const, override));
  MOCK_METHOD(std::vector<std::string>, stream_ids, (), (const, override));
  MOCK_METHOD(std::vector<rtc::scoped_refptr<MediaStreamInterface>>,
              streams,
              (),
              (const, override));
  MOCK_METHOD(cricket::MediaType, media_type, (), (const, override));
  MOCK_METHOD(std::string, id, (), (const, override));
  MOCK_METHOD(RtpParameters, GetParameters, (), (const, override));
  MOCK_METHOD(void, SetObserver, (RtpReceiverObserverInterface*), (override));
  MOCK_METHOD(void,
              SetJitterBufferMinimumDelay,
              (std::optional<double>),
              (override));
  MOCK_METHOD(std::vector<RtpSource>, GetSources, (), (const, override));
  MOCK_METHOD(void,
              SetFrameDecryptor,
              (rtc::scoped_refptr<FrameDecryptorInterface>),
              (override));
  MOCK_METHOD(rtc::scoped_refptr<FrameDecryptorInterface>,
              GetFrameDecryptor,
              (),
              (const, override));

  // RtpReceiverInternal methods.
  MOCK_METHOD(void, Stop, (), (override));
  MOCK_METHOD(void,
              SetMediaChannel,
              (cricket::MediaReceiveChannelInterface*),
              (override));
  MOCK_METHOD(void, SetupMediaChannel, (uint32_t), (override));
  MOCK_METHOD(void, SetupUnsignaledMediaChannel, (), (override));
  MOCK_METHOD(std::optional<uint32_t>, ssrc, (), (const, override));
  MOCK_METHOD(void, NotifyFirstPacketReceived, (), (override));
  MOCK_METHOD(void, set_stream_ids, (std::vector<std::string>), (override));
  MOCK_METHOD(void,
              set_transport,
              (rtc::scoped_refptr<DtlsTransportInterface>),
              (override));
  MOCK_METHOD(void,
              SetStreams,
              (const std::vector<rtc::scoped_refptr<MediaStreamInterface>>&),
              (override));
  MOCK_METHOD(int, AttachmentId, (), (const, override));
};

}  // namespace webrtc

#endif  // PC_TEST_MOCK_RTP_RECEIVER_INTERNAL_H_
