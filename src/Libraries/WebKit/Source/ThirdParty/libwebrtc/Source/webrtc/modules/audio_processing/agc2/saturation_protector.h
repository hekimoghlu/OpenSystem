/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_SATURATION_PROTECTOR_H_
#define MODULES_AUDIO_PROCESSING_AGC2_SATURATION_PROTECTOR_H_

#include <memory>

namespace webrtc {
class ApmDataDumper;

// Saturation protector. Analyzes peak levels and recommends a headroom to
// reduce the chances of clipping.
class SaturationProtector {
 public:
  virtual ~SaturationProtector() = default;

  // Returns the recommended headroom in dB.
  virtual float HeadroomDb() = 0;

  // Analyzes the peak level of a 10 ms frame along with its speech probability
  // and the current speech level estimate to update the recommended headroom.
  virtual void Analyze(float speech_probability,
                       float peak_dbfs,
                       float speech_level_dbfs) = 0;

  // Resets the internal state.
  virtual void Reset() = 0;
};

// Creates a saturation protector that starts at `initial_headroom_db`.
std::unique_ptr<SaturationProtector> CreateSaturationProtector(
    float initial_headroom_db,
    int adjacent_speech_frames_threshold,
    ApmDataDumper* apm_data_dumper);

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_SATURATION_PROTECTOR_H_
