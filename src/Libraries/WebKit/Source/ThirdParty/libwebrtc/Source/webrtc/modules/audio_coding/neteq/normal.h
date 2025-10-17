/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_NORMAL_H_
#define MODULES_AUDIO_CODING_NETEQ_NORMAL_H_

#include <stdint.h>
#include <string.h>  // Access to size_t.

#include "api/neteq/neteq.h"
#include "modules/audio_coding/neteq/statistics_calculator.h"
#include "rtc_base/checks.h"
#include "rtc_base/numerics/safe_conversions.h"

namespace webrtc {

// Forward declarations.
class AudioMultiVector;
class BackgroundNoise;
class DecoderDatabase;
class Expand;

// This class provides the "Normal" DSP operation, that is performed when
// there is no data loss, no need to stretch the timing of the signal, and
// no other "special circumstances" are at hand.
class Normal {
 public:
  Normal(int fs_hz,
         DecoderDatabase* decoder_database,
         const BackgroundNoise& background_noise,
         Expand* expand,
         StatisticsCalculator* statistics)
      : fs_hz_(fs_hz),
        decoder_database_(decoder_database),
        background_noise_(background_noise),
        expand_(expand),
        samples_per_ms_(rtc::CheckedDivExact(fs_hz_, 1000)),
        default_win_slope_Q14_(
            rtc::dchecked_cast<uint16_t>((1 << 14) / samples_per_ms_)),
        statistics_(statistics) {}

  virtual ~Normal() {}

  Normal(const Normal&) = delete;
  Normal& operator=(const Normal&) = delete;

  // Performs the "Normal" operation. The decoder data is supplied in `input`,
  // having `length` samples in total for all channels (interleaved). The
  // result is written to `output`. The number of channels allocated in
  // `output` defines the number of channels that will be used when
  // de-interleaving `input`. `last_mode` contains the mode used in the previous
  // GetAudio call (i.e., not the current one).
  int Process(const int16_t* input,
              size_t length,
              NetEq::Mode last_mode,
              AudioMultiVector* output);

 private:
  int fs_hz_;
  DecoderDatabase* decoder_database_;
  const BackgroundNoise& background_noise_;
  Expand* expand_;
  const size_t samples_per_ms_;
  const int16_t default_win_slope_Q14_;
  StatisticsCalculator* const statistics_;
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_NORMAL_H_
