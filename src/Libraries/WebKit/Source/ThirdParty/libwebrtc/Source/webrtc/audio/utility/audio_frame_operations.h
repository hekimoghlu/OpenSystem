/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 8, 2022.
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
#ifndef AUDIO_UTILITY_AUDIO_FRAME_OPERATIONS_H_
#define AUDIO_UTILITY_AUDIO_FRAME_OPERATIONS_H_

#include <stddef.h>
#include <stdint.h>

#include "absl/base/attributes.h"
#include "api/array_view.h"
#include "api/audio/audio_frame.h"

namespace webrtc {

// TODO(andrew): consolidate this with utility.h and audio_frame_manipulator.h.
// Change reference parameters to pointers. Consider using a namespace rather
// than a class.
class AudioFrameOperations {
 public:
  // Downmixes 4 channels `src_audio` to stereo `dst_audio`. This is an in-place
  // operation, meaning `src_audio` and `dst_audio` may point to the same
  // buffer.
  static void QuadToStereo(InterleavedView<const int16_t> src_audio,
                           InterleavedView<int16_t> dst_audio);

  // `frame.num_channels_` will be updated. This version checks that
  // `num_channels_` is 4 channels.
  static int QuadToStereo(AudioFrame* frame);

  // Downmixes `src_channels` `src_audio` to `dst_channels` `dst_audio`.
  // This is an in-place operation, meaning `src_audio` and `dst_audio`
  // may point to the same buffer. Supported channel combinations are
  // Stereo to Mono, Quad to Mono, and Quad to Stereo.
  static void DownmixChannels(InterleavedView<const int16_t> src_audio,
                              InterleavedView<int16_t> dst_audio);

  // `frame.num_channels_` will be updated. This version checks that
  // `num_channels_` and `dst_channels` are valid and performs relevant downmix.
  // Supported channel combinations are N channels to Mono, and Quad to Stereo.
  static void DownmixChannels(size_t dst_channels, AudioFrame* frame);

  // `frame.num_channels_` will be updated. This version checks that
  // `num_channels_` and `dst_channels` are valid and performs relevant
  // downmix. Supported channel combinations are Mono to N
  // channels. The single channel is replicated.
  static void UpmixChannels(size_t target_number_of_channels,
                            AudioFrame* frame);

  // Swap the left and right channels of `frame`. Fails silently if `frame` is
  // not stereo.
  static void SwapStereoChannels(AudioFrame* frame);

  // Conditionally zero out contents of `frame` for implementing audio mute:
  //  `previous_frame_muted` &&  `current_frame_muted` - Zero out whole frame.
  //  `previous_frame_muted` && !`current_frame_muted` - Fade-in at frame start.
  // !`previous_frame_muted` &&  `current_frame_muted` - Fade-out at frame end.
  // !`previous_frame_muted` && !`current_frame_muted` - Leave frame untouched.
  static void Mute(AudioFrame* frame,
                   bool previous_frame_muted,
                   bool current_frame_muted);

  // Zero out contents of frame.
  static void Mute(AudioFrame* frame);

  static int ScaleWithSat(float scale, AudioFrame* frame);
};

}  // namespace webrtc

#endif  // AUDIO_UTILITY_AUDIO_FRAME_OPERATIONS_H_
