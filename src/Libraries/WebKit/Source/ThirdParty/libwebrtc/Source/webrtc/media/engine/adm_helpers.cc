/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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
#include "media/engine/adm_helpers.h"

#include "api/audio/audio_device.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {
namespace adm_helpers {

// On Windows Vista and newer, Microsoft introduced the concept of "Default
// Communications Device". This means that there are two types of default
// devices (old Wave Audio style default and Default Communications Device).
//
// On Windows systems which only support Wave Audio style default, uses either
// -1 or 0 to select the default device.
//
// Using a #define for AUDIO_DEVICE since we will call *different* versions of
// the ADM functions, depending on the ID type.
#if defined(WEBRTC_WIN)
#define AUDIO_DEVICE_ID \
  (AudioDeviceModule::WindowsDeviceType::kDefaultCommunicationDevice)
#else
#define AUDIO_DEVICE_ID (0u)
#endif  // defined(WEBRTC_WIN)

void Init(AudioDeviceModule* adm) {
  RTC_DCHECK(adm);

  RTC_CHECK_EQ(0, adm->Init()) << "Failed to initialize the ADM.";

  // Playout device.
  {
    if (adm->SetPlayoutDevice(AUDIO_DEVICE_ID) != 0) {
      RTC_LOG(LS_ERROR) << "Unable to set playout device.";
      return;
    }
    if (adm->InitSpeaker() != 0) {
      RTC_LOG(LS_ERROR) << "Unable to access speaker.";
    }

    // Set number of channels
    bool available = false;
    if (adm->StereoPlayoutIsAvailable(&available) != 0) {
      RTC_LOG(LS_ERROR) << "Failed to query stereo playout.";
    }
    if (adm->SetStereoPlayout(available) != 0) {
      RTC_LOG(LS_ERROR) << "Failed to set stereo playout mode.";
    }
  }

  // Recording device.
  {
    if (adm->SetRecordingDevice(AUDIO_DEVICE_ID) != 0) {
      RTC_LOG(LS_ERROR) << "Unable to set recording device.";
      return;
    }
    if (adm->InitMicrophone() != 0) {
      RTC_LOG(LS_ERROR) << "Unable to access microphone.";
    }

    // Set number of channels
    bool available = false;
    if (adm->StereoRecordingIsAvailable(&available) != 0) {
      RTC_LOG(LS_ERROR) << "Failed to query stereo recording.";
    }
    if (adm->SetStereoRecording(available) != 0) {
      RTC_LOG(LS_ERROR) << "Failed to set stereo recording mode.";
    }
  }
}
}  // namespace adm_helpers
}  // namespace webrtc
