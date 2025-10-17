/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 6, 2023.
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
// This file contains the class RtpFormatVp8TestHelper. The class is
// responsible for setting up a fake VP8 bitstream according to the
// RTPVideoHeaderVP8 header. The packetizer can then be provided to this helper
// class, which will then extract all packets and compare to the expected
// outcome.

#ifndef MODULES_RTP_RTCP_SOURCE_RTP_FORMAT_VP8_TEST_HELPER_H_
#define MODULES_RTP_RTCP_SOURCE_RTP_FORMAT_VP8_TEST_HELPER_H_

#include "api/array_view.h"
#include "modules/rtp_rtcp/source/rtp_format_vp8.h"
#include "modules/video_coding/codecs/vp8/include/vp8_globals.h"
#include "rtc_base/buffer.h"

namespace webrtc {

class RtpFormatVp8TestHelper {
 public:
  RtpFormatVp8TestHelper(const RTPVideoHeaderVP8* hdr, size_t payload_len);
  ~RtpFormatVp8TestHelper();

  RtpFormatVp8TestHelper(const RtpFormatVp8TestHelper&) = delete;
  RtpFormatVp8TestHelper& operator=(const RtpFormatVp8TestHelper&) = delete;

  void GetAllPacketsAndCheck(RtpPacketizerVp8* packetizer,
                             rtc::ArrayView<const size_t> expected_sizes);

  rtc::ArrayView<const uint8_t> payload() const { return payload_; }
  size_t payload_size() const { return payload_.size(); }

 private:
  // Returns header size, i.e. payload offset.
  int CheckHeader(rtc::ArrayView<const uint8_t> rtp_payload, bool first);
  void CheckPictureID(rtc::ArrayView<const uint8_t> rtp_payload, int* offset);
  void CheckTl0PicIdx(rtc::ArrayView<const uint8_t> rtp_payload, int* offset);
  void CheckTIDAndKeyIdx(rtc::ArrayView<const uint8_t> rtp_payload,
                         int* offset);
  void CheckPayload(const uint8_t* data_ptr);

  const RTPVideoHeaderVP8* const hdr_info_;
  rtc::Buffer payload_;
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_RTP_FORMAT_VP8_TEST_HELPER_H_
