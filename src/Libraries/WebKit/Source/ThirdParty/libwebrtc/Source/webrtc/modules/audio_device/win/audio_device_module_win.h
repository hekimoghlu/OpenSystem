/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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
#ifndef MODULES_AUDIO_DEVICE_WIN_AUDIO_DEVICE_MODULE_WIN_H_
#define MODULES_AUDIO_DEVICE_WIN_AUDIO_DEVICE_MODULE_WIN_H_

#include <memory>
#include <string>

#include "api/audio/audio_device.h"
#include "api/scoped_refptr.h"
#include "api/task_queue/task_queue_factory.h"

namespace webrtc {

class AudioDeviceBuffer;

namespace webrtc_win {

// This interface represents the main input-related parts of the complete
// AudioDeviceModule interface.
class AudioInput {
 public:
  virtual ~AudioInput() {}

  virtual int Init() = 0;
  virtual int Terminate() = 0;
  virtual int NumDevices() const = 0;
  virtual int SetDevice(int index) = 0;
  virtual int SetDevice(AudioDeviceModule::WindowsDeviceType device) = 0;
  virtual int DeviceName(int index, std::string* name, std::string* guid) = 0;
  virtual void AttachAudioBuffer(AudioDeviceBuffer* audio_buffer) = 0;
  virtual bool RecordingIsInitialized() const = 0;
  virtual int InitRecording() = 0;
  virtual int StartRecording() = 0;
  virtual int StopRecording() = 0;
  virtual bool Recording() = 0;
  virtual int VolumeIsAvailable(bool* available) = 0;
  virtual int RestartRecording() = 0;
  virtual bool Restarting() const = 0;
  virtual int SetSampleRate(uint32_t sample_rate) = 0;
};

// This interface represents the main output-related parts of the complete
// AudioDeviceModule interface.
class AudioOutput {
 public:
  virtual ~AudioOutput() {}

  virtual int Init() = 0;
  virtual int Terminate() = 0;
  virtual int NumDevices() const = 0;
  virtual int SetDevice(int index) = 0;
  virtual int SetDevice(AudioDeviceModule::WindowsDeviceType device) = 0;
  virtual int DeviceName(int index, std::string* name, std::string* guid) = 0;
  virtual void AttachAudioBuffer(AudioDeviceBuffer* audio_buffer) = 0;
  virtual bool PlayoutIsInitialized() const = 0;
  virtual int InitPlayout() = 0;
  virtual int StartPlayout() = 0;
  virtual int StopPlayout() = 0;
  virtual bool Playing() = 0;
  virtual int VolumeIsAvailable(bool* available) = 0;
  virtual int RestartPlayout() = 0;
  virtual bool Restarting() const = 0;
  virtual int SetSampleRate(uint32_t sample_rate) = 0;
};

// Combines an AudioInput and an AudioOutput implementation to build an
// AudioDeviceModule. Hides most parts of the full ADM interface.
rtc::scoped_refptr<AudioDeviceModuleForTest>
CreateWindowsCoreAudioAudioDeviceModuleFromInputAndOutput(
    std::unique_ptr<AudioInput> audio_input,
    std::unique_ptr<AudioOutput> audio_output,
    TaskQueueFactory* task_queue_factory);

}  // namespace webrtc_win

}  // namespace webrtc

#endif  // MODULES_AUDIO_DEVICE_WIN_AUDIO_DEVICE_MODULE_WIN_H_
