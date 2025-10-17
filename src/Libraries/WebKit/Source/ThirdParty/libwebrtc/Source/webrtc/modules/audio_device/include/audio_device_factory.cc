/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 4, 2025.
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
#include "modules/audio_device/include/audio_device_factory.h"

#include <memory>

#if defined(WEBRTC_WIN)
#include "modules/audio_device/win/audio_device_module_win.h"
#include "modules/audio_device/win/core_audio_input_win.h"
#include "modules/audio_device/win/core_audio_output_win.h"
#include "modules/audio_device/win/core_audio_utility_win.h"
#endif

#include "api/task_queue/task_queue_factory.h"
#include "rtc_base/logging.h"

namespace webrtc {

rtc::scoped_refptr<AudioDeviceModule> CreateWindowsCoreAudioAudioDeviceModule(
    TaskQueueFactory* task_queue_factory,
    bool automatic_restart) {
  RTC_DLOG(LS_INFO) << __FUNCTION__;
  return CreateWindowsCoreAudioAudioDeviceModuleForTest(task_queue_factory,
                                                        automatic_restart);
}

rtc::scoped_refptr<AudioDeviceModuleForTest>
CreateWindowsCoreAudioAudioDeviceModuleForTest(
    TaskQueueFactory* task_queue_factory,
    bool automatic_restart) {
  RTC_DLOG(LS_INFO) << __FUNCTION__;
  // Returns NULL if Core Audio is not supported or if COM has not been
  // initialized correctly using ScopedCOMInitializer.
  if (!webrtc_win::core_audio_utility::IsSupported()) {
    RTC_LOG(LS_ERROR)
        << "Unable to create ADM since Core Audio is not supported";
    return nullptr;
  }
  return CreateWindowsCoreAudioAudioDeviceModuleFromInputAndOutput(
      std::make_unique<webrtc_win::CoreAudioInput>(automatic_restart),
      std::make_unique<webrtc_win::CoreAudioOutput>(automatic_restart),
      task_queue_factory);
}

}  // namespace webrtc
