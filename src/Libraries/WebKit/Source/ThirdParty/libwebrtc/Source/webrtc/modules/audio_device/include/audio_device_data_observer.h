/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 13, 2022.
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
#ifndef MODULES_AUDIO_DEVICE_INCLUDE_AUDIO_DEVICE_DATA_OBSERVER_H_
#define MODULES_AUDIO_DEVICE_INCLUDE_AUDIO_DEVICE_DATA_OBSERVER_H_

#include <stddef.h>
#include <stdint.h>

#include "absl/base/attributes.h"
#include "api/audio/audio_device.h"
#include "api/scoped_refptr.h"
#include "api/task_queue/task_queue_factory.h"

namespace webrtc {

// This interface will capture the raw PCM data of both the local captured as
// well as the mixed/rendered remote audio.
class AudioDeviceDataObserver {
 public:
  virtual void OnCaptureData(const void* audio_samples,
                             size_t num_samples,
                             size_t bytes_per_sample,
                             size_t num_channels,
                             uint32_t samples_per_sec) = 0;

  virtual void OnRenderData(const void* audio_samples,
                            size_t num_samples,
                            size_t bytes_per_sample,
                            size_t num_channels,
                            uint32_t samples_per_sec) = 0;

  AudioDeviceDataObserver() = default;
  virtual ~AudioDeviceDataObserver() = default;
};

// Creates an ADMWrapper around an ADM instance that registers
// the provided AudioDeviceDataObserver.
rtc::scoped_refptr<AudioDeviceModule> CreateAudioDeviceWithDataObserver(
    rtc::scoped_refptr<AudioDeviceModule> impl,
    std::unique_ptr<AudioDeviceDataObserver> observer);

// Creates an ADMWrapper around an ADM instance that registers
// the provided AudioDeviceDataObserver.
ABSL_DEPRECATED("")
rtc::scoped_refptr<AudioDeviceModule> CreateAudioDeviceWithDataObserver(
    rtc::scoped_refptr<AudioDeviceModule> impl,
    AudioDeviceDataObserver* observer);

// Creates an ADM instance with AudioDeviceDataObserver registered.
rtc::scoped_refptr<AudioDeviceModule> CreateAudioDeviceWithDataObserver(
    AudioDeviceModule::AudioLayer audio_layer,
    TaskQueueFactory* task_queue_factory,
    std::unique_ptr<AudioDeviceDataObserver> observer);

// Creates an ADM instance with AudioDeviceDataObserver registered.
ABSL_DEPRECATED("")
rtc::scoped_refptr<AudioDeviceModule> CreateAudioDeviceWithDataObserver(
    AudioDeviceModule::AudioLayer audio_layer,
    TaskQueueFactory* task_queue_factory,
    AudioDeviceDataObserver* observer);

}  // namespace webrtc

#endif  // MODULES_AUDIO_DEVICE_INCLUDE_AUDIO_DEVICE_DATA_OBSERVER_H_
