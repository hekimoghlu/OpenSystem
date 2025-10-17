/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 9, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_INCLUDE_AUDIO_FRAME_PROXIES_H_
#define MODULES_AUDIO_PROCESSING_INCLUDE_AUDIO_FRAME_PROXIES_H_

namespace webrtc {

class AudioFrame;
class AudioProcessing;

// Processes a 10 ms `frame` of the primary audio stream using the provided
// AudioProcessing object. On the client-side, this is the near-end (or
// captured) audio. The `sample_rate_hz_`, `num_channels_`, and
// `samples_per_channel_` members of `frame` must be valid. If changed from the
// previous call to this function, it will trigger an initialization of the
// provided AudioProcessing object.
// The function returns any error codes passed from the AudioProcessing
// ProcessStream method.
int ProcessAudioFrame(AudioProcessing* ap, AudioFrame* frame);

// Processes a 10 ms `frame` of the reverse direction audio stream using the
// provided AudioProcessing object. The frame may be modified. On the
// client-side, this is the far-end (or to be rendered) audio. The
// `sample_rate_hz_`, `num_channels_`, and `samples_per_channel_` members of
// `frame` must be valid. If changed from the previous call to this function, it
// will trigger an initialization of the provided AudioProcessing object.
// The function returns any error codes passed from the AudioProcessing
// ProcessReverseStream method.
int ProcessReverseAudioFrame(AudioProcessing* ap, AudioFrame* frame);

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_INCLUDE_AUDIO_FRAME_PROXIES_H_
