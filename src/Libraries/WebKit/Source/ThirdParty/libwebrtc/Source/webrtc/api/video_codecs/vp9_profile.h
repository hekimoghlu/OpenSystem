/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 22, 2022.
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
#ifndef API_VIDEO_CODECS_VP9_PROFILE_H_
#define API_VIDEO_CODECS_VP9_PROFILE_H_

#include <optional>
#include <string>

#include "api/rtp_parameters.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// Profile information for VP9 video.
extern RTC_EXPORT const char kVP9FmtpProfileId[];

enum class VP9Profile {
  kProfile0,
  kProfile1,
  kProfile2,
  kProfile3,
};

// Helper functions to convert VP9Profile to std::string. Returns "0" by
// default.
RTC_EXPORT std::string VP9ProfileToString(VP9Profile profile);

// Helper functions to convert std::string to VP9Profile. Returns null if given
// an invalid profile string.
std::optional<VP9Profile> StringToVP9Profile(const std::string& str);

// Parse profile that is represented as a string of single digit contained in an
// SDP key-value map. A default profile(kProfile0) will be returned if the
// profile key is missing. Nothing will be returned if the key is present but
// the string is invalid.
RTC_EXPORT std::optional<VP9Profile> ParseSdpForVP9Profile(
    const CodecParameterMap& params);

// Returns true if the parameters have the same VP9 profile, or neither contains
// VP9 profile.
bool VP9IsSameProfile(const CodecParameterMap& params1,
                      const CodecParameterMap& params2);

}  // namespace webrtc

#endif  // API_VIDEO_CODECS_VP9_PROFILE_H_
