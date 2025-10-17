/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 23, 2025.
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
#ifndef MODULES_AUDIO_DEVICE_INCLUDE_AUDIO_DEVICE_FACTORY_H_
#define MODULES_AUDIO_DEVICE_INCLUDE_AUDIO_DEVICE_FACTORY_H_

#include <memory>

#include "api/audio/audio_device.h"
#include "api/task_queue/task_queue_factory.h"

namespace webrtc {

// Creates an AudioDeviceModule (ADM) for Windows based on the Core Audio API.
// The creating thread must be a COM thread; otherwise nullptr will be returned.
// By default `automatic_restart` is set to true and it results in support for
// automatic restart of audio if e.g. the existing device is removed. If set to
// false, no attempt to restart audio is performed under these conditions.
//
// Example (assuming webrtc namespace):
//
//  public:
//   rtc::scoped_refptr<AudioDeviceModule> CreateAudioDevice() {
//     task_queue_factory_ = CreateDefaultTaskQueueFactory();
//     // Tell COM that this thread shall live in the MTA.
//     com_initializer_ = std::make_unique<ScopedCOMInitializer>(
//         ScopedCOMInitializer::kMTA);
//     if (!com_initializer_->Succeeded()) {
//       return nullptr;
//     }
//     // Create the ADM with support for automatic restart if devices are
//     // unplugged.
//     return CreateWindowsCoreAudioAudioDeviceModule(
//         task_queue_factory_.get());
//   }
//
//   private:
//    std::unique_ptr<ScopedCOMInitializer> com_initializer_;
//    std::unique_ptr<TaskQueueFactory> task_queue_factory_;
//
rtc::scoped_refptr<AudioDeviceModule> CreateWindowsCoreAudioAudioDeviceModule(
    TaskQueueFactory* task_queue_factory,
    bool automatic_restart = true);

rtc::scoped_refptr<AudioDeviceModuleForTest>
CreateWindowsCoreAudioAudioDeviceModuleForTest(
    TaskQueueFactory* task_queue_factory,
    bool automatic_restart = true);

}  // namespace webrtc

#endif  //  MODULES_AUDIO_DEVICE_INCLUDE_AUDIO_DEVICE_FACTORY_H_
