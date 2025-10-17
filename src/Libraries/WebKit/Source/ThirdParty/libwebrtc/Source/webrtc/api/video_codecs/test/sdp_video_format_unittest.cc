/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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
#include "api/video_codecs/sdp_video_format.h"

#include "api/rtp_parameters.h"
#include "media/base/media_constants.h"
#include "test/gtest.h"

namespace webrtc {

typedef SdpVideoFormat Sdp;
typedef CodecParameterMap Params;

TEST(SdpVideoFormatTest, SameCodecNameNoParameters) {
  EXPECT_TRUE(Sdp("H264").IsSameCodec(Sdp("h264")));
  EXPECT_TRUE(Sdp("VP8").IsSameCodec(Sdp("vp8")));
  EXPECT_TRUE(Sdp("VP9").IsSameCodec(Sdp("vp9")));
  EXPECT_TRUE(Sdp("AV1").IsSameCodec(Sdp("Av1")));
#ifdef RTC_ENABLE_H265
  EXPECT_TRUE(Sdp("H265").IsSameCodec(Sdp("h265")));
#endif
}

TEST(SdpVideoFormatTest, DifferentCodecNameNoParameters) {
  EXPECT_FALSE(Sdp("H264").IsSameCodec(Sdp("VP8")));
  EXPECT_FALSE(Sdp("VP8").IsSameCodec(Sdp("VP9")));
  EXPECT_FALSE(Sdp("AV1").IsSameCodec(Sdp("VP8")));
#ifdef RTC_ENABLE_H265
  EXPECT_FALSE(Sdp("H265").IsSameCodec(Sdp("VP8")));
#endif
}

TEST(SdpVideoFormatTest, SameCodecNameSameParameters) {
  EXPECT_TRUE(Sdp("VP9").IsSameCodec(Sdp("VP9", Params{{"profile-id", "0"}})));
  EXPECT_TRUE(Sdp("VP9", Params{{"profile-id", "0"}})
                  .IsSameCodec(Sdp("VP9", Params{{"profile-id", "0"}})));
  EXPECT_TRUE(Sdp("VP9", Params{{"profile-id", "2"}})
                  .IsSameCodec(Sdp("VP9", Params{{"profile-id", "2"}})));
  EXPECT_TRUE(
      Sdp("H264", Params{{"profile-level-id", "42e01f"}})
          .IsSameCodec(Sdp("H264", Params{{"profile-level-id", "42e01f"}})));
  EXPECT_TRUE(
      Sdp("H264", Params{{"profile-level-id", "640c34"}})
          .IsSameCodec(Sdp("H264", Params{{"profile-level-id", "640c34"}})));
  EXPECT_TRUE(Sdp("AV1").IsSameCodec(Sdp("AV1", Params{{"profile", "0"}})));
  EXPECT_TRUE(Sdp("AV1", Params{{"profile", "0"}})
                  .IsSameCodec(Sdp("AV1", Params{{"profile", "0"}})));
  EXPECT_TRUE(Sdp("AV1", Params{{"profile", "2"}})
                  .IsSameCodec(Sdp("AV1", Params{{"profile", "2"}})));
#ifdef RTC_ENABLE_H265
  EXPECT_TRUE(Sdp("H265").IsSameCodec(Sdp(
      "H265",
      Params{{"profile-id", "1"}, {"tier-flag", "0"}, {"level-id", "93"}})));
  EXPECT_TRUE(
      Sdp("H265",
          Params{{"profile-id", "2"}, {"tier-flag", "0"}, {"level-id", "93"}})
          .IsSameCodec(Sdp("H265", Params{{"profile-id", "2"},
                                          {"tier-flag", "0"},
                                          {"level-id", "93"}})));
#endif
}

TEST(SdpVideoFormatTest, SameCodecNameDifferentParameters) {
  EXPECT_FALSE(Sdp("VP9").IsSameCodec(Sdp("VP9", Params{{"profile-id", "2"}})));
  EXPECT_FALSE(Sdp("VP9", Params{{"profile-id", "0"}})
                   .IsSameCodec(Sdp("VP9", Params{{"profile-id", "1"}})));
  EXPECT_FALSE(Sdp("VP9", Params{{"profile-id", "2"}})
                   .IsSameCodec(Sdp("VP9", Params{{"profile-id", "0"}})));
  EXPECT_FALSE(
      Sdp("H264", Params{{"profile-level-id", "42e01f"}})
          .IsSameCodec(Sdp("H264", Params{{"profile-level-id", "640c34"}})));
  EXPECT_FALSE(
      Sdp("H264", Params{{"profile-level-id", "640c34"}})
          .IsSameCodec(Sdp("H264", Params{{"profile-level-id", "42f00b"}})));
  EXPECT_FALSE(Sdp("AV1").IsSameCodec(Sdp("AV1", Params{{"profile", "1"}})));
  EXPECT_FALSE(Sdp("AV1", Params{{"profile", "0"}})
                   .IsSameCodec(Sdp("AV1", Params{{"profile", "1"}})));
  EXPECT_FALSE(Sdp("AV1", Params{{"profile", "1"}})
                   .IsSameCodec(Sdp("AV1", Params{{"profile", "2"}})));
#ifdef RTC_ENABLE_H265
  EXPECT_FALSE(Sdp("H265").IsSameCodec(Sdp(
      "H265",
      Params{{"profile-id", "0"}, {"tier-flag", "0"}, {"level-id", "93"}})));
  EXPECT_FALSE(Sdp("H265").IsSameCodec(Sdp(
      "H265",
      Params{{"profile-id", "1"}, {"tier-flag", "1"}, {"level-id", "93"}})));
  EXPECT_TRUE(Sdp("H265").IsSameCodec(Sdp(
      "H265",
      Params{{"profile-id", "1"}, {"tier-flag", "0"}, {"level-id", "90"}})));
  EXPECT_FALSE(
      Sdp("H265",
          Params{{"profile-id", "2"}, {"tier-flag", "0"}, {"level-id", "93"}})
          .IsSameCodec(Sdp("H265", Params{{"profile-id", "1"},
                                          {"tier-flag", "0"},
                                          {"level-id", "93"}})));
  EXPECT_FALSE(
      Sdp("H265",
          Params{{"profile-id", "1"}, {"tier-flag", "1"}, {"level-id", "120"}})
          .IsSameCodec(Sdp("H265", Params{{"profile-id", "1"},
                                          {"tier-flag", "0"},
                                          {"level-id", "120"}})));
  EXPECT_TRUE(
      Sdp("H265",
          Params{{"profile-id", "1"}, {"tier-flag", "0"}, {"level-id", "93"}})
          .IsSameCodec(Sdp("H265", Params{{"profile-id", "1"},
                                          {"tier-flag", "0"},
                                          {"level-id", "90"}})));
#endif
}

TEST(SdpVideoFormatTest, DifferentCodecNameSameParameters) {
  EXPECT_FALSE(Sdp("VP9", Params{{"profile-id", "0"}})
                   .IsSameCodec(Sdp("H264", Params{{"profile-id", "0"}})));
  EXPECT_FALSE(Sdp("VP9", Params{{"profile-id", "2"}})
                   .IsSameCodec(Sdp("VP8", Params{{"profile-id", "2"}})));
  EXPECT_FALSE(
      Sdp("H264", Params{{"profile-level-id", "42e01f"}})
          .IsSameCodec(Sdp("VP9", Params{{"profile-level-id", "42e01f"}})));
  EXPECT_FALSE(
      Sdp("H264", Params{{"profile-level-id", "640c34"}})
          .IsSameCodec(Sdp("VP8", Params{{"profile-level-id", "640c34"}})));
  EXPECT_FALSE(Sdp("AV1", Params{{"profile", "0"}})
                   .IsSameCodec(Sdp("H264", Params{{"profile", "0"}})));
  EXPECT_FALSE(Sdp("AV1", Params{{"profile", "2"}})
                   .IsSameCodec(Sdp("VP9", Params{{"profile", "2"}})));
#ifdef RTC_ENABLE_H265
  EXPECT_FALSE(Sdp("H265", Params{{"profile-id", "0"}})
                   .IsSameCodec(Sdp("H264", Params{{"profile-id", "0"}})));
  EXPECT_FALSE(Sdp("H265", Params{{"profile-id", "2"}})
                   .IsSameCodec(Sdp("VP9", Params{{"profile-id", "2"}})));
#endif
}

TEST(SdpVideoFormatTest, H264PacketizationMode) {
  // The default packetization mode is 0.
  EXPECT_TRUE(Sdp("H264", Params{{cricket::kH264FmtpPacketizationMode, "0"}})
                  .IsSameCodec(Sdp("H264")));
  EXPECT_FALSE(Sdp("H264", Params{{cricket::kH264FmtpPacketizationMode, "1"}})
                   .IsSameCodec(Sdp("H264")));

  EXPECT_TRUE(
      Sdp("H264", Params{{cricket::kH264FmtpPacketizationMode, "1"}})
          .IsSameCodec(
              Sdp("H264", Params{{cricket::kH264FmtpPacketizationMode, "1"}})));
}
}  // namespace webrtc
