/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
#ifndef TEST_TESTSUPPORT_COPY_TO_FILE_AUDIO_CAPTURER_H_
#define TEST_TESTSUPPORT_COPY_TO_FILE_AUDIO_CAPTURER_H_

#include <memory>
#include <optional>
#include <string>

#include "api/array_view.h"
#include "common_audio/wav_file.h"
#include "modules/audio_device/include/test_audio_device.h"
#include "rtc_base/buffer.h"

namespace webrtc {
namespace test {

// TestAudioDeviceModule::Capturer that will store audio data, captured by
// delegate to the specified output file. Can be used to create a copy of
// generated audio data to be able then to compare it as a reference with
// audio on the TestAudioDeviceModule::Renderer side.
class CopyToFileAudioCapturer : public TestAudioDeviceModule::Capturer {
 public:
  CopyToFileAudioCapturer(
      std::unique_ptr<TestAudioDeviceModule::Capturer> delegate,
      std::string stream_dump_file_name);
  ~CopyToFileAudioCapturer() override;

  int SamplingFrequency() const override;
  int NumChannels() const override;
  bool Capture(rtc::BufferT<int16_t>* buffer) override;

 private:
  std::unique_ptr<TestAudioDeviceModule::Capturer> delegate_;
  std::unique_ptr<WavWriter> wav_writer_;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_TESTSUPPORT_COPY_TO_FILE_AUDIO_CAPTURER_H_
