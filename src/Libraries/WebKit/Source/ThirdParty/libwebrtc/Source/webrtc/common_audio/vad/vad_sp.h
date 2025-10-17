/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
// This file includes specific signal processing tools used in vad_core.c.

#ifndef COMMON_AUDIO_VAD_VAD_SP_H_
#define COMMON_AUDIO_VAD_VAD_SP_H_

#include "common_audio/vad/vad_core.h"

// Downsamples the signal by a factor 2, eg. 32->16 or 16->8.
//
// Inputs:
//      - signal_in     : Input signal.
//      - in_length     : Length of input signal in samples.
//
// Input & Output:
//      - filter_state  : Current filter states of the two all-pass filters. The
//                        `filter_state` is updated after all samples have been
//                        processed.
//
// Output:
//      - signal_out    : Downsampled signal (of length `in_length` / 2).
void WebRtcVad_Downsampling(const int16_t* signal_in,
                            int16_t* signal_out,
                            int32_t* filter_state,
                            size_t in_length);

// Updates and returns the smoothed feature minimum. As minimum we use the
// median of the five smallest feature values in a 100 frames long window.
// As long as `handle->frame_counter` is zero, that is, we haven't received any
// "valid" data, FindMinimum() outputs the default value of 1600.
//
// Inputs:
//      - feature_value : New feature value to update with.
//      - channel       : Channel number.
//
// Input & Output:
//      - handle        : State information of the VAD.
//
// Returns:
//                      : Smoothed minimum value for a moving window.
int16_t WebRtcVad_FindMinimum(VadInstT* handle,
                              int16_t feature_value,
                              int channel);

#endif  // COMMON_AUDIO_VAD_VAD_SP_H_
