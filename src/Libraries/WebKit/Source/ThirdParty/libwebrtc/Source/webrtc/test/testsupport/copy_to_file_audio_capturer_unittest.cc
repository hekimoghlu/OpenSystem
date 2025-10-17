/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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
#include "test/testsupport/copy_to_file_audio_capturer.h"

#include <memory>
#include <utility>

#include "modules/audio_device/include/test_audio_device.h"
#include "test/gtest.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {
namespace test {

class CopyToFileAudioCapturerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    temp_filename_ = webrtc::test::TempFilename(
        webrtc::test::OutputPath(), "copy_to_file_audio_capturer_unittest");
    std::unique_ptr<TestAudioDeviceModule::Capturer> delegate =
        TestAudioDeviceModule::CreatePulsedNoiseCapturer(32000, 48000);
    capturer_ = std::make_unique<CopyToFileAudioCapturer>(std::move(delegate),
                                                          temp_filename_);
  }

  void TearDown() override { ASSERT_EQ(remove(temp_filename_.c_str()), 0); }

  std::unique_ptr<CopyToFileAudioCapturer> capturer_;
  std::string temp_filename_;
};

TEST_F(CopyToFileAudioCapturerTest, Capture) {
  rtc::BufferT<int16_t> expected_buffer;
  ASSERT_TRUE(capturer_->Capture(&expected_buffer));
  ASSERT_TRUE(!expected_buffer.empty());
  // Destruct capturer to close wav file.
  capturer_.reset(nullptr);

  // Read resulted file content with `wav_file_capture` and compare with
  // what was captured.
  std::unique_ptr<TestAudioDeviceModule::Capturer> wav_file_capturer =
      TestAudioDeviceModule::CreateWavFileReader(temp_filename_, 48000);
  rtc::BufferT<int16_t> actual_buffer;
  wav_file_capturer->Capture(&actual_buffer);
  ASSERT_EQ(actual_buffer, expected_buffer);
}

}  // namespace test
}  // namespace webrtc
