/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 4, 2024.
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
#ifndef AUDIO_REMIX_RESAMPLE_H_
#define AUDIO_REMIX_RESAMPLE_H_

#include "api/audio/audio_frame.h"
#include "api/audio/audio_view.h"
#include "common_audio/resampler/include/push_resampler.h"

namespace webrtc {
namespace voe {

// Note: The RemixAndResample methods assume 10ms buffer sizes.

// Upmix or downmix and resample the audio to `dst_frame`. Expects `dst_frame`
// to have its sample rate and channels members set to the desired values.
// Updates the `samples_per_channel_` member accordingly.
//
// This version has an AudioFrame `src_frame` as input and sets the output
// `timestamp_`, `elapsed_time_ms_` and `ntp_time_ms_` members equals to the
// input ones.
void RemixAndResample(const AudioFrame& src_frame,
                      PushResampler<int16_t>* resampler,
                      AudioFrame* dst_frame);

// TODO(tommi): The `sample_rate_hz` argument can probably be removed since it's
// always related to `src_data.samples_per_frame()'.
void RemixAndResample(InterleavedView<const int16_t> src_data,
                      int sample_rate_hz,
                      PushResampler<int16_t>* resampler,
                      AudioFrame* dst_frame);

}  // namespace voe
}  // namespace webrtc

#endif  // AUDIO_REMIX_RESAMPLE_H_
