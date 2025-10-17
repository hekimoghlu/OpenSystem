/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 14, 2025.
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
#include "api/video_codecs/av1_profile.h"

#include <map>
#include <optional>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "api/rtp_parameters.h"
#include "media/base/media_constants.h"
#include "rtc_base/string_to_number.h"

namespace webrtc {

absl::string_view AV1ProfileToString(AV1Profile profile) {
  switch (profile) {
    case AV1Profile::kProfile0:
      return "0";
    case AV1Profile::kProfile1:
      return "1";
    case AV1Profile::kProfile2:
      return "2";
  }
  return "0";
}

std::optional<AV1Profile> StringToAV1Profile(absl::string_view str) {
  const std::optional<int> i = rtc::StringToNumber<int>(str);
  if (!i.has_value())
    return std::nullopt;

  switch (i.value()) {
    case 0:
      return AV1Profile::kProfile0;
    case 1:
      return AV1Profile::kProfile1;
    case 2:
      return AV1Profile::kProfile2;
    default:
      return std::nullopt;
  }
}

std::optional<AV1Profile> ParseSdpForAV1Profile(
    const CodecParameterMap& params) {
  const auto profile_it = params.find(cricket::kAv1FmtpProfile);
  if (profile_it == params.end())
    return AV1Profile::kProfile0;
  const std::string& profile_str = profile_it->second;
  return StringToAV1Profile(profile_str);
}

bool AV1IsSameProfile(const CodecParameterMap& params1,
                      const CodecParameterMap& params2) {
  const std::optional<AV1Profile> profile = ParseSdpForAV1Profile(params1);
  const std::optional<AV1Profile> other_profile =
      ParseSdpForAV1Profile(params2);
  return profile && other_profile && profile == other_profile;
}

}  // namespace webrtc
