/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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
#include "api/video_codecs/vp9_profile.h"

#include <map>
#include <optional>
#include <string>
#include <utility>

#include "api/rtp_parameters.h"
#include "rtc_base/string_to_number.h"

namespace webrtc {

// Profile information for VP9 video.
const char kVP9FmtpProfileId[] = "profile-id";

std::string VP9ProfileToString(VP9Profile profile) {
  switch (profile) {
    case VP9Profile::kProfile0:
      return "0";
    case VP9Profile::kProfile1:
      return "1";
    case VP9Profile::kProfile2:
      return "2";
    case VP9Profile::kProfile3:
      return "3";
  }
  return "0";
}

std::optional<VP9Profile> StringToVP9Profile(const std::string& str) {
  const std::optional<int> i = rtc::StringToNumber<int>(str);
  if (!i.has_value())
    return std::nullopt;

  switch (i.value()) {
    case 0:
      return VP9Profile::kProfile0;
    case 1:
      return VP9Profile::kProfile1;
    case 2:
      return VP9Profile::kProfile2;
    case 3:
      return VP9Profile::kProfile3;
    default:
      return std::nullopt;
  }
}

std::optional<VP9Profile> ParseSdpForVP9Profile(
    const CodecParameterMap& params) {
  const auto profile_it = params.find(kVP9FmtpProfileId);
  if (profile_it == params.end())
    return VP9Profile::kProfile0;
  const std::string& profile_str = profile_it->second;
  return StringToVP9Profile(profile_str);
}

bool VP9IsSameProfile(const CodecParameterMap& params1,
                      const CodecParameterMap& params2) {
  const std::optional<VP9Profile> profile = ParseSdpForVP9Profile(params1);
  const std::optional<VP9Profile> other_profile =
      ParseSdpForVP9Profile(params2);
  return profile && other_profile && profile == other_profile;
}

}  // namespace webrtc
