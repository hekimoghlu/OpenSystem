/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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
#ifndef API_VIDEO_CODECS_H264_PROFILE_LEVEL_ID_H_
#define API_VIDEO_CODECS_H264_PROFILE_LEVEL_ID_H_

#include <optional>
#include <string>

#include "api/rtp_parameters.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

enum class H264Profile {
  kProfileConstrainedBaseline,
  kProfileBaseline,
  kProfileMain,
  kProfileConstrainedHigh,
  kProfileHigh,
  kProfilePredictiveHigh444,
};

// All values are equal to ten times the level number, except level 1b which is
// special.
enum class H264Level {
  kLevel1_b = 0,
  kLevel1 = 10,
  kLevel1_1 = 11,
  kLevel1_2 = 12,
  kLevel1_3 = 13,
  kLevel2 = 20,
  kLevel2_1 = 21,
  kLevel2_2 = 22,
  kLevel3 = 30,
  kLevel3_1 = 31,
  kLevel3_2 = 32,
  kLevel4 = 40,
  kLevel4_1 = 41,
  kLevel4_2 = 42,
  kLevel5 = 50,
  kLevel5_1 = 51,
  kLevel5_2 = 52
};

struct H264ProfileLevelId {
  constexpr H264ProfileLevelId(H264Profile profile, H264Level level)
      : profile(profile), level(level) {}
  H264Profile profile;
  H264Level level;
};

// Parse profile level id that is represented as a string of 3 hex bytes.
// Nothing will be returned if the string is not a recognized H264
// profile level id.
std::optional<H264ProfileLevelId> ParseH264ProfileLevelId(const char* str);

// Parse profile level id that is represented as a string of 3 hex bytes
// contained in an SDP key-value map. A default profile level id will be
// returned if the profile-level-id key is missing. Nothing will be returned if
// the key is present but the string is invalid.
RTC_EXPORT std::optional<H264ProfileLevelId> ParseSdpForH264ProfileLevelId(
    const CodecParameterMap& params);

// Given that a decoder supports up to a given frame size (in pixels) at up to a
// given number of frames per second, return the highest H.264 level where it
// can guarantee that it will be able to support all valid encoded streams that
// are within that level.
RTC_EXPORT std::optional<H264Level> H264SupportedLevel(
    int max_frame_pixel_count,
    float max_fps);

// Returns canonical string representation as three hex bytes of the profile
// level id, or returns nothing for invalid profile level ids.
RTC_EXPORT std::optional<std::string> H264ProfileLevelIdToString(
    const H264ProfileLevelId& profile_level_id);

// Returns true if the parameters have the same H264 profile (Baseline, High,
// etc).
RTC_EXPORT bool H264IsSameProfile(const CodecParameterMap& params1,
                                  const CodecParameterMap& params2);

}  // namespace webrtc

#endif  // API_VIDEO_CODECS_H264_PROFILE_LEVEL_ID_H_
