/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 12, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_TEST_AUDIO_BUFFER_TOOLS_H_
#define MODULES_AUDIO_PROCESSING_TEST_AUDIO_BUFFER_TOOLS_H_

#include <vector>

#include "api/array_view.h"
#include "api/audio/audio_processing.h"
#include "modules/audio_processing/audio_buffer.h"

namespace webrtc {
namespace test {

// Copies a vector into an audiobuffer.
void CopyVectorToAudioBuffer(const StreamConfig& stream_config,
                             rtc::ArrayView<const float> source,
                             AudioBuffer* destination);

// Extracts a vector from an audiobuffer.
void ExtractVectorFromAudioBuffer(const StreamConfig& stream_config,
                                  AudioBuffer* source,
                                  std::vector<float>* destination);

// Sets all values in `audio_buffer` to `value`.
void FillBuffer(float value, AudioBuffer& audio_buffer);

// Sets all values channel `channel` for `audio_buffer` to `value`.
void FillBufferChannel(float value, int channel, AudioBuffer& audio_buffer);

}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_TEST_AUDIO_BUFFER_TOOLS_H_
