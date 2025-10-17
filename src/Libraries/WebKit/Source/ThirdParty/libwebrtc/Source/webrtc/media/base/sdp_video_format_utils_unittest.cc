/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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
#include "media/base/sdp_video_format_utils.h"

#include <string.h>

#include <map>
#include <utility>

#include "rtc_base/string_to_number.h"
#include "test/gtest.h"

namespace webrtc {
namespace {
// Max frame rate for VP8 and VP9 video.
const char kVPxFmtpMaxFrameRate[] = "max-fr";
// Max frame size for VP8 and VP9 video.
const char kVPxFmtpMaxFrameSize[] = "max-fs";
// Nonstandard per-layer PLI for video.
const char kCodecParamPerLayerPictureLossIndication[] =
    "x-google-per-layer-pli";
}  // namespace

TEST(SdpVideoFormatUtilsTest, TestH264GenerateProfileLevelIdForAnswerEmpty) {
  CodecParameterMap answer_params;
  H264GenerateProfileLevelIdForAnswer(CodecParameterMap(), CodecParameterMap(),
                                      &answer_params);
  EXPECT_TRUE(answer_params.empty());
}

TEST(SdpVideoFormatUtilsTest,
     TestH264GenerateProfileLevelIdForAnswerLevelSymmetryCapped) {
  CodecParameterMap low_level;
  low_level["profile-level-id"] = "42e015";
  CodecParameterMap high_level;
  high_level["profile-level-id"] = "42e01f";

  // Level asymmetry is not allowed; test that answer level is the lower of the
  // local and remote levels.
  CodecParameterMap answer_params;
  H264GenerateProfileLevelIdForAnswer(low_level /* local_supported */,
                                      high_level /* remote_offered */,
                                      &answer_params);
  EXPECT_EQ("42e015", answer_params["profile-level-id"]);

  CodecParameterMap answer_params2;
  H264GenerateProfileLevelIdForAnswer(high_level /* local_supported */,
                                      low_level /* remote_offered */,
                                      &answer_params2);
  EXPECT_EQ("42e015", answer_params2["profile-level-id"]);
}

TEST(SdpVideoFormatUtilsTest,
     TestH264GenerateProfileLevelIdForAnswerConstrainedBaselineLevelAsymmetry) {
  CodecParameterMap local_params;
  local_params["profile-level-id"] = "42e01f";
  local_params["level-asymmetry-allowed"] = "1";
  CodecParameterMap remote_params;
  remote_params["profile-level-id"] = "42e015";
  remote_params["level-asymmetry-allowed"] = "1";
  CodecParameterMap answer_params;
  H264GenerateProfileLevelIdForAnswer(local_params, remote_params,
                                      &answer_params);
  // When level asymmetry is allowed, we can answer a higher level than what was
  // offered.
  EXPECT_EQ("42e01f", answer_params["profile-level-id"]);
}

#ifdef RTC_ENABLE_H265
// Answer should not include explicit PTL info if neither local nor remote set
// any of them.
TEST(SdpVideoFormatUtilsTest, H265GenerateProfileTierLevelEmpty) {
  CodecParameterMap answer_params;
  H265GenerateProfileTierLevelForAnswer(CodecParameterMap(),
                                        CodecParameterMap(), &answer_params);
  EXPECT_TRUE(answer_params.empty());
}

// Answer must use the minimum level as supported by both local and remote.
TEST(SdpVideoFormatUtilsTest, H265GenerateProfileTierLevelNoEmpty) {
  constexpr char kLocallySupportedLevelId[] = "93";
  constexpr char kRemoteOfferedLevelId[] = "120";

  CodecParameterMap local_params;
  local_params["profile-id"] = "1";
  local_params["tier-flag"] = "0";
  local_params["level-id"] = kLocallySupportedLevelId;
  CodecParameterMap remote_params;
  remote_params["profile-id"] = "1";
  remote_params["tier-flag"] = "0";
  remote_params["level-id"] = kRemoteOfferedLevelId;
  CodecParameterMap answer_params;
  H265GenerateProfileTierLevelForAnswer(local_params, remote_params,
                                        &answer_params);
  EXPECT_EQ(kLocallySupportedLevelId, answer_params["level-id"]);
}
#endif

TEST(SdpVideoFormatUtilsTest, MaxFrameRateIsMissingOrInvalid) {
  CodecParameterMap params;
  std::optional<int> empty = ParseSdpForVPxMaxFrameRate(params);
  EXPECT_FALSE(empty);
  params[kVPxFmtpMaxFrameRate] = "-1";
  EXPECT_FALSE(ParseSdpForVPxMaxFrameRate(params));
  params[kVPxFmtpMaxFrameRate] = "0";
  EXPECT_FALSE(ParseSdpForVPxMaxFrameRate(params));
  params[kVPxFmtpMaxFrameRate] = "abcde";
  EXPECT_FALSE(ParseSdpForVPxMaxFrameRate(params));
}

TEST(SdpVideoFormatUtilsTest, MaxFrameRateIsSpecified) {
  CodecParameterMap params;
  params[kVPxFmtpMaxFrameRate] = "30";
  EXPECT_EQ(ParseSdpForVPxMaxFrameRate(params), 30);
  params[kVPxFmtpMaxFrameRate] = "60";
  EXPECT_EQ(ParseSdpForVPxMaxFrameRate(params), 60);
}

TEST(SdpVideoFormatUtilsTest, MaxFrameSizeIsMissingOrInvalid) {
  CodecParameterMap params;
  std::optional<int> empty = ParseSdpForVPxMaxFrameSize(params);
  EXPECT_FALSE(empty);
  params[kVPxFmtpMaxFrameSize] = "-1";
  EXPECT_FALSE(ParseSdpForVPxMaxFrameSize(params));
  params[kVPxFmtpMaxFrameSize] = "0";
  EXPECT_FALSE(ParseSdpForVPxMaxFrameSize(params));
  params[kVPxFmtpMaxFrameSize] = "abcde";
  EXPECT_FALSE(ParseSdpForVPxMaxFrameSize(params));
}

TEST(SdpVideoFormatUtilsTest, MaxFrameSizeIsSpecified) {
  CodecParameterMap params;
  params[kVPxFmtpMaxFrameSize] = "8100";  // 1920 x 1080 / (16^2)
  EXPECT_EQ(ParseSdpForVPxMaxFrameSize(params), 1920 * 1080);
  params[kVPxFmtpMaxFrameSize] = "32400";  // 3840 x 2160 / (16^2)
  EXPECT_EQ(ParseSdpForVPxMaxFrameSize(params), 3840 * 2160);
}

TEST(SdpVideoFormatUtilsTest, PerLayerPictureLossIndication) {
  CodecParameterMap params;
  EXPECT_FALSE(SupportsPerLayerPictureLossIndication(params));
  params[kCodecParamPerLayerPictureLossIndication] = "wrong";
  EXPECT_FALSE(SupportsPerLayerPictureLossIndication(params));
  params[kCodecParamPerLayerPictureLossIndication] = "0";
  EXPECT_FALSE(SupportsPerLayerPictureLossIndication(params));
  params[kCodecParamPerLayerPictureLossIndication] = "1";
  EXPECT_TRUE(SupportsPerLayerPictureLossIndication(params));
}

}  // namespace webrtc
