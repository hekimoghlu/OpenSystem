/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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
#include "modules/audio_processing/test/runtime_setting_util.h"

#include "rtc_base/checks.h"

namespace webrtc {

void ReplayRuntimeSetting(AudioProcessing* apm,
                          const webrtc::audioproc::RuntimeSetting& setting) {
  RTC_CHECK(apm);
  // TODO(bugs.webrtc.org/9138): Add ability to handle different types
  // of settings. Currently CapturePreGain, CaptureFixedPostGain and
  // PlayoutVolumeChange are supported.
  RTC_CHECK(setting.has_capture_pre_gain() ||
            setting.has_capture_fixed_post_gain() ||
            setting.has_playout_volume_change());

  if (setting.has_capture_pre_gain()) {
    apm->SetRuntimeSetting(
        AudioProcessing::RuntimeSetting::CreateCapturePreGain(
            setting.capture_pre_gain()));
  } else if (setting.has_capture_fixed_post_gain()) {
    apm->SetRuntimeSetting(
        AudioProcessing::RuntimeSetting::CreateCaptureFixedPostGain(
            setting.capture_fixed_post_gain()));
  } else if (setting.has_playout_volume_change()) {
    apm->SetRuntimeSetting(
        AudioProcessing::RuntimeSetting::CreatePlayoutVolumeChange(
            setting.playout_volume_change()));
  } else if (setting.has_playout_audio_device_change()) {
    apm->SetRuntimeSetting(
        AudioProcessing::RuntimeSetting::CreatePlayoutAudioDeviceChange(
            {setting.playout_audio_device_change().id(),
             setting.playout_audio_device_change().max_volume()}));
  } else if (setting.has_capture_output_used()) {
    apm->SetRuntimeSetting(
        AudioProcessing::RuntimeSetting::CreateCaptureOutputUsedSetting(
            setting.capture_output_used()));
  }
}
}  // namespace webrtc
