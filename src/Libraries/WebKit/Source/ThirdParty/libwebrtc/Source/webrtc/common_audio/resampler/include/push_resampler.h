/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 9, 2022.
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
#ifndef COMMON_AUDIO_RESAMPLER_INCLUDE_PUSH_RESAMPLER_H_
#define COMMON_AUDIO_RESAMPLER_INCLUDE_PUSH_RESAMPLER_H_

#include <memory>
#include <vector>

#include "api/audio/audio_view.h"

namespace webrtc {

class PushSincResampler;

// Wraps PushSincResampler to provide stereo support.
// Note: This implementation assumes 10ms buffer sizes throughout.
template <typename T>
class PushResampler final {
 public:
  PushResampler();
  PushResampler(size_t src_samples_per_channel,
                size_t dst_samples_per_channel,
                size_t num_channels);
  ~PushResampler();

  // Returns the total number of samples provided in destination (e.g. 32 kHz,
  // 2 channel audio gives 640 samples).
  int Resample(InterleavedView<const T> src, InterleavedView<T> dst);
  // For when a deinterleaved/mono channel already exists and we can skip the
  // deinterleaved operation.
  int Resample(MonoView<const T> src, MonoView<T> dst);

 private:
  // Ensures that source and destination buffers for deinterleaving are
  // correctly configured prior to resampling that requires deinterleaving.
  void EnsureInitialized(size_t src_samples_per_channel,
                         size_t dst_samples_per_channel,
                         size_t num_channels);

  // Buffers used for when a deinterleaving step is necessary.
  std::unique_ptr<T[]> source_;
  std::unique_ptr<T[]> destination_;
  DeinterleavedView<T> source_view_;
  DeinterleavedView<T> destination_view_;

  std::vector<std::unique_ptr<PushSincResampler>> resamplers_;
};
}  // namespace webrtc

#endif  // COMMON_AUDIO_RESAMPLER_INCLUDE_PUSH_RESAMPLER_H_
