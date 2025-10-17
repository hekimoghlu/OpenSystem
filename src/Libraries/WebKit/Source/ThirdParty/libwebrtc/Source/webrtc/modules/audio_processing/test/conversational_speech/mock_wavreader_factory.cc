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
#include "modules/audio_processing/test/conversational_speech/mock_wavreader_factory.h"

#include "absl/strings/string_view.h"
#include "modules/audio_processing/test/conversational_speech/mock_wavreader.h"
#include "rtc_base/logging.h"
#include "test/gmock.h"

namespace webrtc {
namespace test {
namespace conversational_speech {

using ::testing::_;
using ::testing::Invoke;

MockWavReaderFactory::MockWavReaderFactory(
    const Params& default_params,
    const std::map<std::string, const Params>& params)
    : default_params_(default_params), audiotrack_names_params_(params) {
  ON_CALL(*this, Create(_))
      .WillByDefault(Invoke(this, &MockWavReaderFactory::CreateMock));
}

MockWavReaderFactory::MockWavReaderFactory(const Params& default_params)
    : MockWavReaderFactory(default_params,
                           std::map<std::string, const Params>{}) {}

MockWavReaderFactory::~MockWavReaderFactory() = default;

std::unique_ptr<WavReaderInterface> MockWavReaderFactory::CreateMock(
    absl::string_view filepath) {
  // Search the parameters corresponding to filepath.
  size_t delimiter = filepath.find_last_of("/\\");  // Either windows or posix
  std::string filename(filepath.substr(
      delimiter == absl::string_view::npos ? 0 : delimiter + 1));
  const auto it = audiotrack_names_params_.find(filename);

  // If not found, use default parameters.
  if (it == audiotrack_names_params_.end()) {
    RTC_LOG(LS_VERBOSE) << "using default parameters for " << filepath;
    return std::unique_ptr<WavReaderInterface>(new MockWavReader(
        default_params_.sample_rate, default_params_.num_channels,
        default_params_.num_samples));
  }

  // Found, use the audiotrack-specific parameters.
  RTC_LOG(LS_VERBOSE) << "using ad-hoc parameters for " << filepath;
  RTC_LOG(LS_VERBOSE) << "sample_rate " << it->second.sample_rate;
  RTC_LOG(LS_VERBOSE) << "num_channels " << it->second.num_channels;
  RTC_LOG(LS_VERBOSE) << "num_samples " << it->second.num_samples;
  return std::unique_ptr<WavReaderInterface>(new MockWavReader(
      it->second.sample_rate, it->second.num_channels, it->second.num_samples));
}

}  // namespace conversational_speech
}  // namespace test
}  // namespace webrtc
