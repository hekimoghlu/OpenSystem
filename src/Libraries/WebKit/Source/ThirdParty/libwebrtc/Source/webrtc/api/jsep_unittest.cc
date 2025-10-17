/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 15, 2024.
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
#include "api/jsep.h"

#include <memory>
#include <string>

#include "absl/strings/str_cat.h"
#include "rtc_base/logging.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {

using ::testing::HasSubstr;

TEST(JsepTest, AbslStringifySdp) {
  std::string sdp =
      "v=0\r\n"
      "o=- 0 3 IN IP4 127.0.0.1\r\n"
      "s=-\r\n"
      "t=0 0\r\n"
      "a=group:BUNDLE 0 1\r\n"
      "a=fingerprint:sha-1 "
      "4A:AD:B9:B1:3F:82:18:3B:54:02:12:DF:3E:5D:49:6B:19:E5:7C:AB\r\n"
      "a=setup:actpass\r\n"
      "a=ice-ufrag:ETEn\r\n"
      "a=ice-pwd:OtSK0WpNtpUjkY4+86js7Z/l\r\n"
      "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
      "c=IN IP4 0.0.0.0\r\n"
      "a=rtcp-mux\r\n"
      "a=sendonly\r\n"
      "a=mid:0\r\n"
      "a=rtpmap:111 opus/48000/2\r\n"
      "m=video 9 UDP/TLS/RTP/SAVPF 111\r\n"
      "c=IN IP4 0.0.0.0\r\n"
      "a=rtcp-mux\r\n"
      "a=sendonly\r\n"
      "a=mid:1\r\n"
      "a=rtpmap:111 H264/90000\r\n"
      "a=fmtp:111 "
      "level-asymmetry-allowed=1;packetization-mode=0;profile-level-id="
      "42e01f\r\n";

  std::unique_ptr<SessionDescriptionInterface> some_sdp =
      CreateSessionDescription(SdpType::kOffer, sdp);
  // Verify that sending the SDP to the log compiles.
  RTC_LOG(LS_VERBOSE) << "The SDP is " << *some_sdp;
  // Since create/stringify mangles order of fields, we only test
  // some substrings.
  EXPECT_THAT(absl::StrCat(*some_sdp), HasSubstr("a=rtpmap:111 opus/48000"));
  EXPECT_THAT(
      absl::StrCat(*some_sdp),
      HasSubstr(
          "a=fingerprint:sha-1 "
          "4A:AD:B9:B1:3F:82:18:3B:54:02:12:DF:3E:5D:49:6B:19:E5:7C:AB\r\n"));
}

}  // namespace webrtc
