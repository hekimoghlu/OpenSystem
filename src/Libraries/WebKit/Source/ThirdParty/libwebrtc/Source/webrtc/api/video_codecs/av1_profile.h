/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 13, 2023.
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
#ifndef API_VIDEO_CODECS_AV1_PROFILE_H_
#define API_VIDEO_CODECS_AV1_PROFILE_H_

#include <optional>

#include "absl/strings/string_view.h"
#include "api/rtp_parameters.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// Profiles can be found at:
// https://aomedia.org/av1/specification/annex-a/#profiles
// The enum values match the number specified in the SDP.
enum class AV1Profile {
  kProfile0 = 0,
  kProfile1 = 1,
  kProfile2 = 2,
};

// Helper function which converts an AV1Profile to std::string. Returns "0" if
// an unknown value is passed in.
RTC_EXPORT absl::string_view AV1ProfileToString(AV1Profile profile);

// Helper function which converts a std::string to AV1Profile. Returns null if
// |profile| is not a valid profile string.
std::optional<AV1Profile> StringToAV1Profile(absl::string_view profile);

// Parses an SDP key-value map of format parameters to retrive an AV1 profile.
// Returns an AV1Profile if one has been specified, `kProfile0` if no profile is
// specified and an empty value if the profile key is present but contains an
// invalid value.
RTC_EXPORT std::optional<AV1Profile> ParseSdpForAV1Profile(
    const CodecParameterMap& params);

// Returns true if the parameters have the same AV1 profile or neither contains
// an AV1 profile, otherwise false.
bool AV1IsSameProfile(const CodecParameterMap& params1,
                      const CodecParameterMap& params2);

}  // namespace webrtc

#endif  // API_VIDEO_CODECS_AV1_PROFILE_H_
