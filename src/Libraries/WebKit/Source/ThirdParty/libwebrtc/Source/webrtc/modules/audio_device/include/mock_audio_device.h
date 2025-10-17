/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 5, 2024.
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
#ifndef MODULES_AUDIO_DEVICE_INCLUDE_MOCK_AUDIO_DEVICE_H_
#define MODULES_AUDIO_DEVICE_INCLUDE_MOCK_AUDIO_DEVICE_H_

#include <string>

#include "api/audio/audio_device.h"
#include "api/make_ref_counted.h"
#include "test/gmock.h"

namespace webrtc {
namespace test {

class MockAudioDeviceModule : public AudioDeviceModule {
 public:
  static rtc::scoped_refptr<MockAudioDeviceModule> CreateNice() {
    return rtc::make_ref_counted<::testing::NiceMock<MockAudioDeviceModule>>();
  }
  static rtc::scoped_refptr<MockAudioDeviceModule> CreateStrict() {
    return rtc::make_ref_counted<
        ::testing::StrictMock<MockAudioDeviceModule>>();
  }

  // AudioDeviceModule.
  MOCK_METHOD(int32_t,
              ActiveAudioLayer,
              (AudioLayer * audioLayer),
              (const, override));
  MOCK_METHOD(int32_t,
              RegisterAudioCallback,
              (AudioTransport * audioCallback),
              (override));
  MOCK_METHOD(int32_t, Init, (), (override));
  MOCK_METHOD(int32_t, Terminate, (), (override));
  MOCK_METHOD(bool, Initialized, (), (const, override));
  MOCK_METHOD(int16_t, PlayoutDevices, (), (override));
  MOCK_METHOD(int16_t, RecordingDevices, (), (override));
  MOCK_METHOD(int32_t,
              PlayoutDeviceName,
              (uint16_t index,
               char name[kAdmMaxDeviceNameSize],
               char guid[kAdmMaxGuidSize]),
              (override));
  MOCK_METHOD(int32_t,
              RecordingDeviceName,
              (uint16_t index,
               char name[kAdmMaxDeviceNameSize],
               char guid[kAdmMaxGuidSize]),
              (override));
  MOCK_METHOD(int32_t, SetPlayoutDevice, (uint16_t index), (override));
  MOCK_METHOD(int32_t,
              SetPlayoutDevice,
              (WindowsDeviceType device),
              (override));
  MOCK_METHOD(int32_t, SetRecordingDevice, (uint16_t index), (override));
  MOCK_METHOD(int32_t,
              SetRecordingDevice,
              (WindowsDeviceType device),
              (override));
  MOCK_METHOD(int32_t, PlayoutIsAvailable, (bool* available), (override));
  MOCK_METHOD(int32_t, InitPlayout, (), (override));
  MOCK_METHOD(bool, PlayoutIsInitialized, (), (const, override));
  MOCK_METHOD(int32_t, RecordingIsAvailable, (bool* available), (override));
  MOCK_METHOD(int32_t, InitRecording, (), (override));
  MOCK_METHOD(bool, RecordingIsInitialized, (), (const, override));
  MOCK_METHOD(int32_t, StartPlayout, (), (override));
  MOCK_METHOD(int32_t, StopPlayout, (), (override));
  MOCK_METHOD(bool, Playing, (), (const, override));
  MOCK_METHOD(int32_t, StartRecording, (), (override));
  MOCK_METHOD(int32_t, StopRecording, (), (override));
  MOCK_METHOD(bool, Recording, (), (const, override));
  MOCK_METHOD(int32_t, InitSpeaker, (), (override));
  MOCK_METHOD(bool, SpeakerIsInitialized, (), (const, override));
  MOCK_METHOD(int32_t, InitMicrophone, (), (override));
  MOCK_METHOD(bool, MicrophoneIsInitialized, (), (const, override));
  MOCK_METHOD(int32_t, SpeakerVolumeIsAvailable, (bool* available), (override));
  MOCK_METHOD(int32_t, SetSpeakerVolume, (uint32_t volume), (override));
  MOCK_METHOD(int32_t, SpeakerVolume, (uint32_t * volume), (const, override));
  MOCK_METHOD(int32_t,
              MaxSpeakerVolume,
              (uint32_t * maxVolume),
              (const, override));
  MOCK_METHOD(int32_t,
              MinSpeakerVolume,
              (uint32_t * minVolume),
              (const, override));
  MOCK_METHOD(int32_t,
              MicrophoneVolumeIsAvailable,
              (bool* available),
              (override));
  MOCK_METHOD(int32_t, SetMicrophoneVolume, (uint32_t volume), (override));
  MOCK_METHOD(int32_t,
              MicrophoneVolume,
              (uint32_t * volume),
              (const, override));
  MOCK_METHOD(int32_t,
              MaxMicrophoneVolume,
              (uint32_t * maxVolume),
              (const, override));
  MOCK_METHOD(int32_t,
              MinMicrophoneVolume,
              (uint32_t * minVolume),
              (const, override));
  MOCK_METHOD(int32_t, SpeakerMuteIsAvailable, (bool* available), (override));
  MOCK_METHOD(int32_t, SetSpeakerMute, (bool enable), (override));
  MOCK_METHOD(int32_t, SpeakerMute, (bool* enabled), (const, override));
  MOCK_METHOD(int32_t,
              MicrophoneMuteIsAvailable,
              (bool* available),
              (override));
  MOCK_METHOD(int32_t, SetMicrophoneMute, (bool enable), (override));
  MOCK_METHOD(int32_t, MicrophoneMute, (bool* enabled), (const, override));
  MOCK_METHOD(int32_t,
              StereoPlayoutIsAvailable,
              (bool* available),
              (const, override));
  MOCK_METHOD(int32_t, SetStereoPlayout, (bool enable), (override));
  MOCK_METHOD(int32_t, StereoPlayout, (bool* enabled), (const, override));
  MOCK_METHOD(int32_t,
              StereoRecordingIsAvailable,
              (bool* available),
              (const, override));
  MOCK_METHOD(int32_t, SetStereoRecording, (bool enable), (override));
  MOCK_METHOD(int32_t, StereoRecording, (bool* enabled), (const, override));
  MOCK_METHOD(int32_t, PlayoutDelay, (uint16_t * delayMS), (const, override));
  MOCK_METHOD(bool, BuiltInAECIsAvailable, (), (const, override));
  MOCK_METHOD(bool, BuiltInAGCIsAvailable, (), (const, override));
  MOCK_METHOD(bool, BuiltInNSIsAvailable, (), (const, override));
  MOCK_METHOD(int32_t, EnableBuiltInAEC, (bool enable), (override));
  MOCK_METHOD(int32_t, EnableBuiltInAGC, (bool enable), (override));
  MOCK_METHOD(int32_t, EnableBuiltInNS, (bool enable), (override));
  MOCK_METHOD(int32_t, GetPlayoutUnderrunCount, (), (const, override));
#if defined(WEBRTC_IOS)
  MOCK_METHOD(int,
              GetPlayoutAudioParameters,
              (AudioParameters * params),
              (const, override));
  MOCK_METHOD(int,
              GetRecordAudioParameters,
              (AudioParameters * params),
              (const, override));
#endif  // WEBRTC_IOS
};
}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_DEVICE_INCLUDE_MOCK_AUDIO_DEVICE_H_
