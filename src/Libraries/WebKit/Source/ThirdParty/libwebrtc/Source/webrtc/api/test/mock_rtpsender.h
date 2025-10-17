/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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
#ifndef API_TEST_MOCK_RTPSENDER_H_
#define API_TEST_MOCK_RTPSENDER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "api/crypto/frame_encryptor_interface.h"
#include "api/dtls_transport_interface.h"
#include "api/dtmf_sender_interface.h"
#include "api/frame_transformer_interface.h"
#include "api/make_ref_counted.h"
#include "api/media_stream_interface.h"
#include "api/media_types.h"
#include "api/rtc_error.h"
#include "api/rtp_parameters.h"
#include "api/rtp_sender_interface.h"
#include "api/scoped_refptr.h"
#include "api/video_codecs/video_encoder_factory.h"
#include "rtc_base/ref_counted_object.h"
#include "test/gmock.h"

namespace webrtc {

class MockRtpSender : public RtpSenderInterface {
 public:
  static rtc::scoped_refptr<MockRtpSender> Create() {
    return rtc::make_ref_counted<MockRtpSender>();
  }

  MOCK_METHOD(bool, SetTrack, (MediaStreamTrackInterface*), (override));
  MOCK_METHOD(rtc::scoped_refptr<MediaStreamTrackInterface>,
              track,
              (),
              (const, override));
  MOCK_METHOD(rtc::scoped_refptr<DtlsTransportInterface>, dtls_transport, (),
              (const, override));
  MOCK_METHOD(uint32_t, ssrc, (), (const, override));
  MOCK_METHOD(cricket::MediaType, media_type, (), (const, override));
  MOCK_METHOD(std::string, id, (), (const, override));
  MOCK_METHOD(std::vector<std::string>, stream_ids, (), (const, override));
  MOCK_METHOD(void, SetStreams, (const std::vector<std::string>&), (override));
  MOCK_METHOD(std::vector<RtpEncodingParameters>,
              init_send_encodings,
              (),
              (const, override));
  MOCK_METHOD(RtpParameters, GetParameters, (), (const, override));
  MOCK_METHOD(RTCError, SetParameters, (const RtpParameters&), (override));
  MOCK_METHOD(void,
              SetParametersAsync,
              (const RtpParameters&, SetParametersCallback),
              (override));
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
};

static_assert(!std::is_abstract_v<rtc::RefCountedObject<MockRtpSender>>, "");
}  // namespace webrtc

#endif  // API_TEST_MOCK_RTPSENDER_H_
