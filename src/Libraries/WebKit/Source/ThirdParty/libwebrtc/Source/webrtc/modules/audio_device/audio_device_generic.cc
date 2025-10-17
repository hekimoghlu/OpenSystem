/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 30, 2025.
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
#include "modules/audio_device/audio_device_generic.h"

#include "rtc_base/logging.h"

namespace webrtc {

bool AudioDeviceGeneric::BuiltInAECIsAvailable() const {
  RTC_LOG_F(LS_ERROR) << "Not supported on this platform";
  return false;
}

int32_t AudioDeviceGeneric::EnableBuiltInAEC(bool /* enable */) {
  RTC_LOG_F(LS_ERROR) << "Not supported on this platform";
  return -1;
}

bool AudioDeviceGeneric::BuiltInAGCIsAvailable() const {
  RTC_LOG_F(LS_ERROR) << "Not supported on this platform";
  return false;
}

int32_t AudioDeviceGeneric::EnableBuiltInAGC(bool /* enable */) {
  RTC_LOG_F(LS_ERROR) << "Not supported on this platform";
  return -1;
}

bool AudioDeviceGeneric::BuiltInNSIsAvailable() const {
  RTC_LOG_F(LS_ERROR) << "Not supported on this platform";
  return false;
}

int32_t AudioDeviceGeneric::EnableBuiltInNS(bool /* enable */) {
  RTC_LOG_F(LS_ERROR) << "Not supported on this platform";
  return -1;
}

int32_t AudioDeviceGeneric::GetPlayoutUnderrunCount() const {
  RTC_LOG_F(LS_ERROR) << "Not supported on this platform";
  return -1;
}

#if defined(WEBRTC_IOS)
int AudioDeviceGeneric::GetPlayoutAudioParameters(
    AudioParameters* params) const {
  RTC_LOG_F(LS_ERROR) << "Not supported on this platform";
  return -1;
}

int AudioDeviceGeneric::GetRecordAudioParameters(
    AudioParameters* params) const {
  RTC_LOG_F(LS_ERROR) << "Not supported on this platform";
  return -1;
}
#endif  // WEBRTC_IOS

}  // namespace webrtc
