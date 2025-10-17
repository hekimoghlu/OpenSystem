/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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
#ifndef API_TEST_MOCK_RTPRECEIVER_H_
#define API_TEST_MOCK_RTPRECEIVER_H_

#include <optional>
#include <string>
#include <vector>

#include "api/crypto/frame_decryptor_interface.h"
#include "api/media_stream_interface.h"
#include "api/media_types.h"
#include "api/rtp_parameters.h"
#include "api/rtp_receiver_interface.h"
#include "api/scoped_refptr.h"
#include "api/transport/rtp/rtp_source.h"
#include "rtc_base/ref_counted_object.h"
#include "test/gmock.h"

namespace webrtc {

class MockRtpReceiver : public rtc::RefCountedObject<RtpReceiverInterface> {
 public:
  MOCK_METHOD(rtc::scoped_refptr<MediaStreamTrackInterface>,
              track,
              (),
              (const, override));
  MOCK_METHOD(std::vector<rtc::scoped_refptr<MediaStreamInterface>>,
              streams,
              (),
              (const, override));
  MOCK_METHOD(cricket::MediaType, media_type, (), (const, override));
  MOCK_METHOD(std::string, id, (), (const, override));
  MOCK_METHOD(RtpParameters, GetParameters, (), (const, override));
  MOCK_METHOD(bool,
              SetParameters,
              (const webrtc::RtpParameters& parameters),
              (override));
  MOCK_METHOD(void, SetObserver, (RtpReceiverObserverInterface*), (override));
  MOCK_METHOD(void,
              SetJitterBufferMinimumDelay,
              (std::optional<double>),
              (override));
  MOCK_METHOD(std::vector<RtpSource>, GetSources, (), (const, override));
  MOCK_METHOD(void,
              SetFrameDecryptor,
              (rtc::scoped_refptr<webrtc::FrameDecryptorInterface>),
              (override));
  MOCK_METHOD(rtc::scoped_refptr<webrtc::FrameDecryptorInterface>,
              GetFrameDecryptor,
              (),
              (const, override));
};

}  // namespace webrtc

#endif  // API_TEST_MOCK_RTPRECEIVER_H_
