/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 15, 2025.
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
#ifndef MODULES_VIDEO_CODING_CODECS_VP8_INCLUDE_TEMPORAL_LAYERS_CHECKER_H_
#define MODULES_VIDEO_CODING_CODECS_VP8_INCLUDE_TEMPORAL_LAYERS_CHECKER_H_

#include <stdint.h>

#include <memory>

#include "api/video_codecs/vp8_frame_config.h"
#include "api/video_codecs/vp8_temporal_layers.h"

namespace webrtc {

// Interface for a class that verifies correctness of temporal layer
// configurations (dependencies, sync flag, etc).
// Intended to be used in tests as well as with real apps in debug mode.
class TemporalLayersChecker {
 public:
  explicit TemporalLayersChecker(int num_temporal_layers);
  virtual ~TemporalLayersChecker() {}

  virtual bool CheckTemporalConfig(bool frame_is_keyframe,
                                   const Vp8FrameConfig& frame_config);

  static std::unique_ptr<TemporalLayersChecker> CreateTemporalLayersChecker(
      Vp8TemporalLayersType type,
      int num_temporal_layers);

 private:
  struct BufferState {
    BufferState() : is_keyframe(true), temporal_layer(0), sequence_number(0) {}
    bool is_keyframe;
    uint8_t temporal_layer;
    uint32_t sequence_number;
  };
  bool CheckAndUpdateBufferState(BufferState* state,
                                 bool* need_sync,
                                 bool frame_is_keyframe,
                                 uint8_t temporal_layer,
                                 Vp8FrameConfig::BufferFlags flags,
                                 uint32_t sequence_number,
                                 uint32_t* lowest_sequence_referenced);
  BufferState last_;
  BufferState arf_;
  BufferState golden_;
  int num_temporal_layers_;
  uint32_t sequence_number_;
  uint32_t last_sync_sequence_number_;
  uint32_t last_tl0_sequence_number_;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_CODECS_VP8_INCLUDE_TEMPORAL_LAYERS_CHECKER_H_
