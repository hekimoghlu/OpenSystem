/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 21, 2022.
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
#ifndef AUDIO_DEVICE_AUDIO_MIXER_MANAGER_ALSA_LINUX_H_
#define AUDIO_DEVICE_AUDIO_MIXER_MANAGER_ALSA_LINUX_H_

#include <alsa/asoundlib.h>

#include "api/audio/audio_device.h"
#include "modules/audio_device/linux/alsasymboltable_linux.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {

class AudioMixerManagerLinuxALSA {
 public:
  int32_t OpenSpeaker(char* deviceName) RTC_LOCKS_EXCLUDED(mutex_);
  int32_t OpenMicrophone(char* deviceName) RTC_LOCKS_EXCLUDED(mutex_);
  int32_t SetSpeakerVolume(uint32_t volume) RTC_LOCKS_EXCLUDED(mutex_);
  int32_t SpeakerVolume(uint32_t& volume) const;
  int32_t MaxSpeakerVolume(uint32_t& maxVolume) const;
  int32_t MinSpeakerVolume(uint32_t& minVolume) const;
  int32_t SpeakerVolumeIsAvailable(bool& available);
  int32_t SpeakerMuteIsAvailable(bool& available);
  int32_t SetSpeakerMute(bool enable) RTC_LOCKS_EXCLUDED(mutex_);
  int32_t SpeakerMute(bool& enabled) const;
  int32_t MicrophoneMuteIsAvailable(bool& available);
  int32_t SetMicrophoneMute(bool enable) RTC_LOCKS_EXCLUDED(mutex_);
  int32_t MicrophoneMute(bool& enabled) const;
  int32_t MicrophoneVolumeIsAvailable(bool& available);
  int32_t SetMicrophoneVolume(uint32_t volume) RTC_LOCKS_EXCLUDED(mutex_);
  int32_t MicrophoneVolume(uint32_t& volume) const;
  int32_t MaxMicrophoneVolume(uint32_t& maxVolume) const;
  int32_t MinMicrophoneVolume(uint32_t& minVolume) const;
  int32_t Close() RTC_LOCKS_EXCLUDED(mutex_);
  int32_t CloseSpeaker() RTC_LOCKS_EXCLUDED(mutex_);
  int32_t CloseMicrophone() RTC_LOCKS_EXCLUDED(mutex_);
  bool SpeakerIsInitialized() const;
  bool MicrophoneIsInitialized() const;

 public:
  AudioMixerManagerLinuxALSA();
  ~AudioMixerManagerLinuxALSA();

 private:
  int32_t CloseSpeakerLocked() RTC_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  int32_t CloseMicrophoneLocked() RTC_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  int32_t LoadMicMixerElement() const;
  int32_t LoadSpeakerMixerElement() const;
  void GetControlName(char* controlName, char* deviceName) const;

 private:
  Mutex mutex_;
  mutable snd_mixer_t* _outputMixerHandle;
  char _outputMixerStr[kAdmMaxDeviceNameSize];
  mutable snd_mixer_t* _inputMixerHandle;
  char _inputMixerStr[kAdmMaxDeviceNameSize];
  mutable snd_mixer_elem_t* _outputMixerElement;
  mutable snd_mixer_elem_t* _inputMixerElement;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_DEVICE_MAIN_SOURCE_LINUX_AUDIO_MIXER_MANAGER_ALSA_LINUX_H_
