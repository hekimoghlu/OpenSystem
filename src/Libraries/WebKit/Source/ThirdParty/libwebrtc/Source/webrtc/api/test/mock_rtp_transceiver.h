/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 29, 2024.
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
#ifndef API_TEST_MOCK_RTP_TRANSCEIVER_H_
#define API_TEST_MOCK_RTP_TRANSCEIVER_H_

#include <optional>
#include <string>
#include <vector>

#include "api/array_view.h"
#include "api/make_ref_counted.h"
#include "api/media_types.h"
#include "api/rtc_error.h"
#include "api/rtp_parameters.h"
#include "api/rtp_receiver_interface.h"
#include "api/rtp_sender_interface.h"
#include "api/rtp_transceiver_direction.h"
#include "api/rtp_transceiver_interface.h"
#include "api/scoped_refptr.h"
#include "test/gmock.h"

namespace webrtc {

class MockRtpTransceiver : public RtpTransceiverInterface {
 public:
  MockRtpTransceiver() = default;

  static rtc::scoped_refptr<MockRtpTransceiver> Create() {
    return rtc::make_ref_counted<MockRtpTransceiver>();
  }

  MOCK_METHOD(cricket::MediaType, media_type, (), (const, override));
  MOCK_METHOD(std::optional<std::string>, mid, (), (const, override));
  MOCK_METHOD(rtc::scoped_refptr<RtpSenderInterface>,
              sender,
              (),
              (const, override));
  MOCK_METHOD(rtc::scoped_refptr<RtpReceiverInterface>,
              receiver,
              (),
              (const, override));
  MOCK_METHOD(bool, stopped, (), (const, override));
  MOCK_METHOD(bool, stopping, (), (const, override));
  MOCK_METHOD(RtpTransceiverDirection, direction, (), (const, override));
  MOCK_METHOD(void,
              SetDirection,
              (RtpTransceiverDirection new_direction),
              (override));
  MOCK_METHOD(RTCError,
              SetDirectionWithError,
              (RtpTransceiverDirection new_direction),
              (override));
  MOCK_METHOD(std::optional<RtpTransceiverDirection>,
              current_direction,
              (),
              (const, override));
  MOCK_METHOD(std::optional<RtpTransceiverDirection>,
              fired_direction,
              (),
              (const, override));
  MOCK_METHOD(RTCError, StopStandard, (), (override));
  MOCK_METHOD(void, StopInternal, (), (override));
  MOCK_METHOD(void, Stop, (), (override));
  MOCK_METHOD(RTCError,
              SetCodecPreferences,
              (rtc::ArrayView<RtpCodecCapability> codecs),
              (override));
  MOCK_METHOD(std::vector<RtpCodecCapability>,
              codec_preferences,
              (),
              (const, override));
  MOCK_METHOD(std::vector<RtpHeaderExtensionCapability>,
              GetHeaderExtensionsToNegotiate,
              (),
              (const, override));
  MOCK_METHOD(std::vector<RtpHeaderExtensionCapability>,
              GetNegotiatedHeaderExtensions,
              (),
              (const, override));
  MOCK_METHOD(
      webrtc::RTCError,
      SetHeaderExtensionsToNegotiate,
      (rtc::ArrayView<const RtpHeaderExtensionCapability> header_extensions),
      (override));
};

}  // namespace webrtc

#endif  // API_TEST_MOCK_RTP_TRANSCEIVER_H_
