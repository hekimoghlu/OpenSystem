/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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
#ifndef MODULES_AUDIO_CODING_ACM2_ACM_REMIXING_H_
#define MODULES_AUDIO_CODING_ACM2_ACM_REMIXING_H_

#include <vector>

#include "api/array_view.h"
#include "api/audio/audio_frame.h"

namespace webrtc {

// Stereo-to-mono downmixing. The length of the output must equal to the number
// of samples per channel in the input.
void DownMixFrame(const AudioFrame& input, rtc::ArrayView<int16_t> output);

// Remixes the interleaved input frame to an interleaved output data vector. The
// remixed data replaces the data in the output vector which is resized if
// needed. The remixing supports any combination of input and output channels,
// as well as any number of samples per channel.
void ReMixFrame(const AudioFrame& input,
                size_t num_output_channels,
                std::vector<int16_t>* output);

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_ACM2_ACM_REMIXING_H_
