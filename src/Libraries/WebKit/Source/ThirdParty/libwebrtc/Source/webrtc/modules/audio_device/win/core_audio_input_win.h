/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 24, 2023.
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
#ifndef MODULES_AUDIO_DEVICE_WIN_CORE_AUDIO_INPUT_WIN_H_
#define MODULES_AUDIO_DEVICE_WIN_CORE_AUDIO_INPUT_WIN_H_

#include <memory>
#include <optional>
#include <string>

#include "modules/audio_device/win/audio_device_module_win.h"
#include "modules/audio_device/win/core_audio_base_win.h"

namespace webrtc {

class AudioDeviceBuffer;
class FineAudioBuffer;

namespace webrtc_win {

// Windows specific AudioInput implementation using a CoreAudioBase class where
// an input direction is set at construction. Supports capture device handling
// and streaming of captured audio to a WebRTC client.
class CoreAudioInput final : public CoreAudioBase, public AudioInput {
 public:
  CoreAudioInput(bool automatic_restart);
  ~CoreAudioInput() override;

  // AudioInput implementation.
  int Init() override;
  int Terminate() override;
  int NumDevices() const override;
  int SetDevice(int index) override;
  int SetDevice(AudioDeviceModule::WindowsDeviceType device) override;
  int DeviceName(int index, std::string* name, std::string* guid) override;
  void AttachAudioBuffer(AudioDeviceBuffer* audio_buffer) override;
  bool RecordingIsInitialized() const override;
  int InitRecording() override;
  int StartRecording() override;
  int StopRecording() override;
  bool Recording() override;
  int VolumeIsAvailable(bool* available) override;
  int RestartRecording() override;
  bool Restarting() const override;
  int SetSampleRate(uint32_t sample_rate) override;

  CoreAudioInput(const CoreAudioInput&) = delete;
  CoreAudioInput& operator=(const CoreAudioInput&) = delete;

 private:
  void ReleaseCOMObjects();
  bool OnDataCallback(uint64_t device_frequency);
  bool OnErrorCallback(ErrorType error);
  std::optional<int> EstimateLatencyMillis(uint64_t capture_time_100ns);
  bool HandleStreamDisconnected();

  std::unique_ptr<FineAudioBuffer> fine_audio_buffer_;
  Microsoft::WRL::ComPtr<IAudioCaptureClient> audio_capture_client_;
  std::optional<double> qpc_to_100ns_;
};

}  // namespace webrtc_win

}  // namespace webrtc

#endif  // MODULES_AUDIO_DEVICE_WIN_CORE_AUDIO_INPUT_WIN_H_
