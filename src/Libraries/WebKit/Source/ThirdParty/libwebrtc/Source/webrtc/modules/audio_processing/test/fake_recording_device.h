/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 27, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_TEST_FAKE_RECORDING_DEVICE_H_
#define MODULES_AUDIO_PROCESSING_TEST_FAKE_RECORDING_DEVICE_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "api/array_view.h"
#include "common_audio/channel_buffer.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace test {

class FakeRecordingDeviceWorker;

// Class for simulating a microphone with analog gain.
//
// The intended modes of operation are the following:
//
// FakeRecordingDevice fake_mic(255, 1);
//
// fake_mic.SetMicLevel(170);
// fake_mic.SimulateAnalogGain(buffer);
//
// When the mic level to undo is known:
//
// fake_mic.SetMicLevel(170);
// fake_mic.SetUndoMicLevel(30);
// fake_mic.SimulateAnalogGain(buffer);
//
// The second option virtually restores the unmodified microphone level. Calling
// SimulateAnalogGain() will first "undo" the gain applied by the real
// microphone (e.g., 30).
class FakeRecordingDevice final {
 public:
  FakeRecordingDevice(int initial_mic_level, int device_kind);
  ~FakeRecordingDevice();

  int MicLevel() const;
  void SetMicLevel(int level);
  void SetUndoMicLevel(int level);

  // Simulates the analog gain.
  // If `real_device_level` is a valid level, the unmodified mic signal is
  // virtually restored. To skip the latter step set `real_device_level` to
  // an empty value.
  void SimulateAnalogGain(rtc::ArrayView<int16_t> buffer);

  // Simulates the analog gain.
  // If `real_device_level` is a valid level, the unmodified mic signal is
  // virtually restored. To skip the latter step set `real_device_level` to
  // an empty value.
  void SimulateAnalogGain(ChannelBuffer<float>* buffer);

 private:
  // Fake recording device worker.
  std::unique_ptr<FakeRecordingDeviceWorker> worker_;
};

}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_TEST_FAKE_RECORDING_DEVICE_H_
