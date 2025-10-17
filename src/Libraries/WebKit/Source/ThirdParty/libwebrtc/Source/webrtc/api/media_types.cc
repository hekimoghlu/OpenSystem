/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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
#include "api/media_types.h"

#include <string>

#include "rtc_base/checks.h"

namespace cricket {

const char kMediaTypeVideo[] = "video";
const char kMediaTypeAudio[] = "audio";
const char kMediaTypeData[] = "data";

std::string MediaTypeToString(MediaType type) {
  switch (type) {
    case MEDIA_TYPE_AUDIO:
      return kMediaTypeAudio;
    case MEDIA_TYPE_VIDEO:
      return kMediaTypeVideo;
    case MEDIA_TYPE_DATA:
      return kMediaTypeData;
    case MEDIA_TYPE_UNSUPPORTED:
      // Unsupported media stores the m=<mediatype> differently.
      RTC_DCHECK_NOTREACHED();
      return "";
  }
  RTC_CHECK_NOTREACHED();
}

}  // namespace cricket
