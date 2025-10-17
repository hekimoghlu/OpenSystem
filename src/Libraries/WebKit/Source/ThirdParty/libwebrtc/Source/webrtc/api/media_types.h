/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 17, 2023.
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
#ifndef API_MEDIA_TYPES_H_
#define API_MEDIA_TYPES_H_

#include <string>

#include "rtc_base/system/rtc_export.h"

// The cricket and webrtc have separate definitions for what a media type is.
// They're not compatible. Watch out for this.

namespace cricket {

enum MediaType {
  MEDIA_TYPE_AUDIO,
  MEDIA_TYPE_VIDEO,
  MEDIA_TYPE_DATA,
  MEDIA_TYPE_UNSUPPORTED
};

extern const char kMediaTypeAudio[];
extern const char kMediaTypeVideo[];
extern const char kMediaTypeData[];

RTC_EXPORT std::string MediaTypeToString(MediaType type);

}  // namespace cricket

namespace webrtc {

enum class MediaType { ANY, AUDIO, VIDEO, DATA };

}  // namespace webrtc

#endif  // API_MEDIA_TYPES_H_
