/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 11, 2025.
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
#ifndef MODULES_AUDIO_MIXER_FRAME_COMBINER_H_
#define MODULES_AUDIO_MIXER_FRAME_COMBINER_H_

#include <memory>
#include <vector>

#include "api/array_view.h"
#include "api/audio/audio_frame.h"
#include "modules/audio_processing/agc2/limiter.h"

namespace webrtc {
class ApmDataDumper;

class FrameCombiner {
 public:
  explicit FrameCombiner(bool use_limiter);
  ~FrameCombiner();

  // Combine several frames into one. Assumes sample_rate,
  // samples_per_channel of the input frames match the parameters. The
  // parameters 'number_of_channels' and 'sample_rate' are needed
  // because 'mix_list' can be empty. The parameter
  // 'number_of_streams' is used for determining whether to pass the
  // data through a limiter.
  void Combine(rtc::ArrayView<AudioFrame* const> mix_list,
               size_t number_of_channels,
               int sample_rate,
               size_t number_of_streams,
               AudioFrame* audio_frame_for_mixing);

  // Stereo, 48 kHz, 10 ms.
  static constexpr size_t kMaximumNumberOfChannels = 8;
  static constexpr size_t kMaximumChannelSize = 48 * 10;

 private:
  std::unique_ptr<ApmDataDumper> data_dumper_;
  Limiter limiter_;
  const bool use_limiter_;
  std::array<float, kMaximumChannelSize * kMaximumNumberOfChannels>
      mixing_buffer_ = {};
};
}  // namespace webrtc

#endif  // MODULES_AUDIO_MIXER_FRAME_COMBINER_H_
