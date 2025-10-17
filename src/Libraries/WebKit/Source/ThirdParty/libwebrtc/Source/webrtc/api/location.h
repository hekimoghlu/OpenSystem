/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 30, 2023.
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
#ifndef API_LOCATION_H_
#define API_LOCATION_H_

#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// Location provides basic info where of an object was constructed, or was
// significantly brought to life. This is a stripped down version of
// https://source.chromium.org/chromium/chromium/src/+/main:base/location.h
// that only specifies an interface compatible to how base::Location is
// supposed to be used.
// The declaration is overriden inside the Chromium build.
class RTC_EXPORT Location {
 public:
  static Location Current() { return Location(); }
};

}  // namespace webrtc

#endif  // API_LOCATION_H_
