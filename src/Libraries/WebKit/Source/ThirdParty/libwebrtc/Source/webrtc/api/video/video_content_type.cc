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
#include "api/video/video_content_type.h"

#include <cstdint>

#include "rtc_base/checks.h"

namespace webrtc {
namespace videocontenttypehelpers {

namespace {
static constexpr uint8_t kScreenshareBitsSize = 1;
static constexpr uint8_t kScreenshareBitsMask =
    (1u << kScreenshareBitsSize) - 1;
}  // namespace

bool IsScreenshare(const VideoContentType& content_type) {
  // Ensure no bits apart from the screenshare bit is set.
  // This CHECK is a temporary measure to detect code that introduces
  // values according to old versions.
  RTC_CHECK((static_cast<uint8_t>(content_type) & !kScreenshareBitsMask) == 0);
  return (static_cast<uint8_t>(content_type) & kScreenshareBitsMask) > 0;
}

bool IsValidContentType(uint8_t value) {
  // Only the screenshare bit is allowed.
  // However, due to previous usage of the next 5 bits, we allow
  // the lower 6 bits to be set.
  return value < (1 << 6);
}

const char* ToString(const VideoContentType& content_type) {
  return IsScreenshare(content_type) ? "screen" : "realtime";
}
}  // namespace videocontenttypehelpers
}  // namespace webrtc
