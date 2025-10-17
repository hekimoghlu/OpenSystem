/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 4, 2022.
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
#ifndef MODULES_AUDIO_MIXER_AUDIO_FRAME_MANIPULATOR_H_
#define MODULES_AUDIO_MIXER_AUDIO_FRAME_MANIPULATOR_H_

#include <stddef.h>
#include <stdint.h>

#include "api/audio/audio_frame.h"

namespace webrtc {

// Updates the audioFrame's energy (based on its samples).
uint32_t AudioMixerCalculateEnergy(const AudioFrame& audio_frame);

// Ramps up or down the provided audio frame. Ramp(0, 1, frame) will
// linearly increase the samples in the frame from 0 to full volume.
void Ramp(float start_gain, float target_gain, AudioFrame* audio_frame);

// Downmixes or upmixes a frame between stereo and mono.
void RemixFrame(size_t target_number_of_channels, AudioFrame* frame);

}  // namespace webrtc

#endif  // MODULES_AUDIO_MIXER_AUDIO_FRAME_MANIPULATOR_H_
