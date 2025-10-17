/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 18, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_VECTOR_FLOAT_FRAME_H_
#define MODULES_AUDIO_PROCESSING_AGC2_VECTOR_FLOAT_FRAME_H_

#include <vector>

#include "api/audio/audio_view.h"
#include "modules/audio_processing/include/audio_frame_view.h"

namespace webrtc {

// A construct consisting of a multi-channel audio frame, and a FloatFrame view
// of it.
class VectorFloatFrame {
 public:
  VectorFloatFrame(int num_channels,
                   int samples_per_channel,
                   float start_value);
  ~VectorFloatFrame();

  AudioFrameView<float> float_frame_view();
  AudioFrameView<const float> float_frame_view() const;

  DeinterleavedView<float> view() { return view_; }
  DeinterleavedView<const float> view() const { return view_; }

 private:
  std::vector<float> channels_;
  DeinterleavedView<float> view_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_VECTOR_FLOAT_FRAME_H_
