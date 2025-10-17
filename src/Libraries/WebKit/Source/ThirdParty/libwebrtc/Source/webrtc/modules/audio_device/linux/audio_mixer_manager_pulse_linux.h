/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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
#ifndef AUDIO_DEVICE_AUDIO_MIXER_MANAGER_PULSE_LINUX_H_
#define AUDIO_DEVICE_AUDIO_MIXER_MANAGER_PULSE_LINUX_H_

#include <pulse/pulseaudio.h>
#include <stdint.h>

#include "api/sequence_checker.h"

#ifndef UINT32_MAX
#define UINT32_MAX ((uint32_t)-1)
#endif

namespace webrtc {

class AudioMixerManagerLinuxPulse {
 public:
  int32_t SetPlayStream(pa_stream* playStream);
  int32_t SetRecStream(pa_stream* recStream);
  int32_t OpenSpeaker(uint16_t deviceIndex);
  int32_t OpenMicrophone(uint16_t deviceIndex);
  int32_t SetSpeakerVolume(uint32_t volume);
  int32_t SpeakerVolume(uint32_t& volume) const;
  int32_t MaxSpeakerVolume(uint32_t& maxVolume) const;
  int32_t MinSpeakerVolume(uint32_t& minVolume) const;
  int32_t SpeakerVolumeIsAvailable(bool& available);
  int32_t SpeakerMuteIsAvailable(bool& available);
  int32_t SetSpeakerMute(bool enable);
  int32_t StereoPlayoutIsAvailable(bool& available);
  int32_t StereoRecordingIsAvailable(bool& available);
  int32_t SpeakerMute(bool& enabled) const;
  int32_t MicrophoneMuteIsAvailable(bool& available);
  int32_t SetMicrophoneMute(bool enable);
  int32_t MicrophoneMute(bool& enabled) const;
  int32_t MicrophoneVolumeIsAvailable(bool& available);
  int32_t SetMicrophoneVolume(uint32_t volume);
  int32_t MicrophoneVolume(uint32_t& volume) const;
  int32_t MaxMicrophoneVolume(uint32_t& maxVolume) const;
  int32_t MinMicrophoneVolume(uint32_t& minVolume) const;
  int32_t SetPulseAudioObjects(pa_threaded_mainloop* mainloop,
                               pa_context* context);
  int32_t Close();
  int32_t CloseSpeaker();
  int32_t CloseMicrophone();
  bool SpeakerIsInitialized() const;
  bool MicrophoneIsInitialized() const;

 public:
  AudioMixerManagerLinuxPulse();
  ~AudioMixerManagerLinuxPulse();

 private:
  static void PaSinkInfoCallback(pa_context* c,
                                 const pa_sink_info* i,
                                 int eol,
                                 void* pThis);
  static void PaSinkInputInfoCallback(pa_context* c,
                                      const pa_sink_input_info* i,
                                      int eol,
                                      void* pThis);
  static void PaSourceInfoCallback(pa_context* c,
                                   const pa_source_info* i,
                                   int eol,
                                   void* pThis);
  static void PaSetVolumeCallback(pa_context* /*c*/,
                                  int success,
                                  void* /*pThis*/);
  void PaSinkInfoCallbackHandler(const pa_sink_info* i, int eol);
  void PaSinkInputInfoCallbackHandler(const pa_sink_input_info* i, int eol);
  void PaSourceInfoCallbackHandler(const pa_source_info* i, int eol);

  void WaitForOperationCompletion(pa_operation* paOperation) const;

  bool GetSinkInputInfo() const;
  bool GetSinkInfoByIndex(int device_index) const;
  bool GetSourceInfoByIndex(int device_index) const;

 private:
  int16_t _paOutputDeviceIndex;
  int16_t _paInputDeviceIndex;

  pa_stream* _paPlayStream;
  pa_stream* _paRecStream;

  pa_threaded_mainloop* _paMainloop;
  pa_context* _paContext;

  mutable uint32_t _paVolume;
  mutable uint32_t _paMute;
  mutable uint32_t _paVolSteps;
  bool _paSpeakerMute;
  mutable uint32_t _paSpeakerVolume;
  mutable uint8_t _paChannels;
  bool _paObjectsSet;

  // Stores thread ID in constructor.
  // We can then use RTC_DCHECK_RUN_ON(&worker_thread_checker_) to ensure that
  // other methods are called from the same thread.
  // Currently only does RTC_DCHECK(thread_checker_.IsCurrent()).
  SequenceChecker thread_checker_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_DEVICE_MAIN_SOURCE_LINUX_AUDIO_MIXER_MANAGER_PULSE_LINUX_H_
