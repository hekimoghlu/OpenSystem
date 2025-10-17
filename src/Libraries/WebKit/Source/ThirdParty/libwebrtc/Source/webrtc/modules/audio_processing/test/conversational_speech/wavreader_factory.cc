/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 15, 2022.
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
#include "modules/audio_processing/test/conversational_speech/wavreader_factory.h"

#include <cstddef>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "common_audio/wav_file.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace test {
namespace {

using conversational_speech::WavReaderInterface;

class WavReaderAdaptor final : public WavReaderInterface {
 public:
  explicit WavReaderAdaptor(absl::string_view filepath)
      : wav_reader_(filepath) {}
  ~WavReaderAdaptor() override = default;

  size_t ReadFloatSamples(rtc::ArrayView<float> samples) override {
    return wav_reader_.ReadSamples(samples.size(), samples.begin());
  }

  size_t ReadInt16Samples(rtc::ArrayView<int16_t> samples) override {
    return wav_reader_.ReadSamples(samples.size(), samples.begin());
  }

  int SampleRate() const override { return wav_reader_.sample_rate(); }

  size_t NumChannels() const override { return wav_reader_.num_channels(); }

  size_t NumSamples() const override { return wav_reader_.num_samples(); }

 private:
  WavReader wav_reader_;
};

}  // namespace

namespace conversational_speech {

WavReaderFactory::WavReaderFactory() = default;

WavReaderFactory::~WavReaderFactory() = default;

std::unique_ptr<WavReaderInterface> WavReaderFactory::Create(
    absl::string_view filepath) const {
  return std::unique_ptr<WavReaderAdaptor>(new WavReaderAdaptor(filepath));
}

}  // namespace conversational_speech
}  // namespace test
}  // namespace webrtc
