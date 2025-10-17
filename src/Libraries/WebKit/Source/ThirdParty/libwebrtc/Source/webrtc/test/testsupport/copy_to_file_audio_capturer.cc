/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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

namespace webrtc {
namespace test {

CopyToFileAudioCapturer::CopyToFileAudioCapturer(
    std::unique_ptr<TestAudioDeviceModule::Capturer> delegate,
    std::string stream_dump_file_name)
    : delegate_(std::move(delegate)),
      wav_writer_(std::make_unique<WavWriter>(std::move(stream_dump_file_name),
                                              delegate_->SamplingFrequency(),
                                              delegate_->NumChannels())) {}
CopyToFileAudioCapturer::~CopyToFileAudioCapturer() = default;

int CopyToFileAudioCapturer::SamplingFrequency() const {
  return delegate_->SamplingFrequency();
}

int CopyToFileAudioCapturer::NumChannels() const {
  return delegate_->NumChannels();
}

bool CopyToFileAudioCapturer::Capture(rtc::BufferT<int16_t>* buffer) {
  bool result = delegate_->Capture(buffer);
  if (result) {
    wav_writer_->WriteSamples(buffer->data(), buffer->size());
  }
  return result;
}

}  // namespace test
}  // namespace webrtc
