/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 22, 2024.
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
#ifndef MODULES_AUDIO_PROCESSING_VAD_PITCH_INTERNAL_H_
#define MODULES_AUDIO_PROCESSING_VAD_PITCH_INTERNAL_H_

namespace webrtc {

// TODO(turajs): Write a description of this function. Also be consistent with
// usage of `sampling_rate_hz` vs `kSamplingFreqHz`.
void GetSubframesPitchParameters(int sampling_rate_hz,
                                 double* gains,
                                 double* lags,
                                 int num_in_frames,
                                 int num_out_frames,
                                 double* log_old_gain,
                                 double* old_lag,
                                 double* log_pitch_gain,
                                 double* pitch_lag_hz);

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_VAD_PITCH_INTERNAL_H_
