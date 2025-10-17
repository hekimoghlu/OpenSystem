/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 11, 2021.
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
#include "modules/audio_processing/test/conversational_speech/mock_wavreader.h"

namespace webrtc {
namespace test {
namespace conversational_speech {

using ::testing::Return;

MockWavReader::MockWavReader(int sample_rate,
                             size_t num_channels,
                             size_t num_samples)
    : sample_rate_(sample_rate),
      num_channels_(num_channels),
      num_samples_(num_samples) {
  ON_CALL(*this, SampleRate()).WillByDefault(Return(sample_rate_));
  ON_CALL(*this, NumChannels()).WillByDefault(Return(num_channels_));
  ON_CALL(*this, NumSamples()).WillByDefault(Return(num_samples_));
}

MockWavReader::~MockWavReader() = default;

}  // namespace conversational_speech
}  // namespace test
}  // namespace webrtc
