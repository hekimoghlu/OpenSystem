/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 17, 2022.
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
#ifndef API_VIDEO_CODECS_SCALABILITY_MODE_H_
#define API_VIDEO_CODECS_SCALABILITY_MODE_H_

#include <stddef.h>
#include <stdint.h>

#include "absl/strings/string_view.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// Supported scalability modes. Most applications should use the
// PeerConnection-level apis where scalability mode is represented as a string.
// This list of currently recognized modes is intended for the api boundary
// between webrtc and injected encoders. Any application usage outside of
// injected encoders is strongly discouraged.
enum class ScalabilityMode : uint8_t {
  kL1T1,
  kL1T2,
  kL1T3,
  kL2T1,
  kL2T1h,
  kL2T1_KEY,
  kL2T2,
  kL2T2h,
  kL2T2_KEY,
  kL2T2_KEY_SHIFT,
  kL2T3,
  kL2T3h,
  kL2T3_KEY,
  kL3T1,
  kL3T1h,
  kL3T1_KEY,
  kL3T2,
  kL3T2h,
  kL3T2_KEY,
  kL3T3,
  kL3T3h,
  kL3T3_KEY,
  kS2T1,
  kS2T1h,
  kS2T2,
  kS2T2h,
  kS2T3,
  kS2T3h,
  kS3T1,
  kS3T1h,
  kS3T2,
  kS3T2h,
  kS3T3,
  kS3T3h,
};

inline constexpr ScalabilityMode kAllScalabilityModes[] = {
    // clang-format off
    ScalabilityMode::kL1T1,
    ScalabilityMode::kL1T2,
    ScalabilityMode::kL1T3,
    ScalabilityMode::kL2T1,
    ScalabilityMode::kL2T1h,
    ScalabilityMode::kL2T1_KEY,
    ScalabilityMode::kL2T2,
    ScalabilityMode::kL2T2h,
    ScalabilityMode::kL2T2_KEY,
    ScalabilityMode::kL2T2_KEY_SHIFT,
    ScalabilityMode::kL2T3,
    ScalabilityMode::kL2T3h,
    ScalabilityMode::kL2T3_KEY,
    ScalabilityMode::kL3T1,
    ScalabilityMode::kL3T1h,
    ScalabilityMode::kL3T1_KEY,
    ScalabilityMode::kL3T2,
    ScalabilityMode::kL3T2h,
    ScalabilityMode::kL3T2_KEY,
    ScalabilityMode::kL3T3,
    ScalabilityMode::kL3T3h,
    ScalabilityMode::kL3T3_KEY,
    ScalabilityMode::kS2T1,
    ScalabilityMode::kS2T1h,
    ScalabilityMode::kS2T2,
    ScalabilityMode::kS2T2h,
    ScalabilityMode::kS2T3,
    ScalabilityMode::kS2T3h,
    ScalabilityMode::kS3T1,
    ScalabilityMode::kS3T1h,
    ScalabilityMode::kS3T2,
    ScalabilityMode::kS3T2h,
    ScalabilityMode::kS3T3,
    ScalabilityMode::kS3T3h,
    // clang-format on
};

inline constexpr size_t kScalabilityModeCount =
    sizeof(kAllScalabilityModes) / sizeof(ScalabilityMode);

RTC_EXPORT
absl::string_view ScalabilityModeToString(ScalabilityMode scalability_mode);

}  // namespace webrtc

#endif  // API_VIDEO_CODECS_SCALABILITY_MODE_H_
