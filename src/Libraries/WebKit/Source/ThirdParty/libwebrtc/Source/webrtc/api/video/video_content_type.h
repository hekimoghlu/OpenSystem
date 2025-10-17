/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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
#ifndef API_VIDEO_VIDEO_CONTENT_TYPE_H_
#define API_VIDEO_VIDEO_CONTENT_TYPE_H_

#include <stdint.h>

namespace webrtc {

// VideoContentType stored as a single byte, which is sent over the network
// in the rtp-hdrext/video-content-type extension.
// Only the lowest bit is used, per the enum.
enum class VideoContentType : uint8_t {
  UNSPECIFIED = 0,
  SCREENSHARE = 1,
};

namespace videocontenttypehelpers {
bool IsScreenshare(const VideoContentType& content_type);

bool IsValidContentType(uint8_t value);

const char* ToString(const VideoContentType& content_type);
}  // namespace videocontenttypehelpers

}  // namespace webrtc

#endif  // API_VIDEO_VIDEO_CONTENT_TYPE_H_
