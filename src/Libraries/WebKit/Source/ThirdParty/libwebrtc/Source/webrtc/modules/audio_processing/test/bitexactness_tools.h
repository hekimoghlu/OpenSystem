/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 12, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_TEST_BITEXACTNESS_TOOLS_H_
#define MODULES_AUDIO_PROCESSING_TEST_BITEXACTNESS_TOOLS_H_

#include <string>

#include "api/array_view.h"
#include "modules/audio_coding/neteq/tools/input_audio_file.h"
#include "test/gtest.h"

namespace webrtc {
namespace test {

// Returns test vector to use for the render signal in an
// APM bitexactness test.
std::string GetApmRenderTestVectorFileName(int sample_rate_hz);

// Returns test vector to use for the capture signal in an
// APM bitexactness test.
std::string GetApmCaptureTestVectorFileName(int sample_rate_hz);

// Extract float samples of up to two channels from a pcm file.
void ReadFloatSamplesFromStereoFile(size_t samples_per_channel,
                                    size_t num_channels,
                                    InputAudioFile* stereo_pcm_file,
                                    rtc::ArrayView<float> data);

// Verifies a frame against a reference and returns the results as an
// AssertionResult.
::testing::AssertionResult VerifyDeinterleavedArray(
    size_t samples_per_channel,
    size_t num_channels,
    rtc::ArrayView<const float> reference,
    rtc::ArrayView<const float> output,
    float element_error_bound);

// Verifies a vector against a reference and returns the results as an
// AssertionResult.
::testing::AssertionResult VerifyArray(rtc::ArrayView<const float> reference,
                                       rtc::ArrayView<const float> output,
                                       float element_error_bound);

}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_TEST_BITEXACTNESS_TOOLS_H_
