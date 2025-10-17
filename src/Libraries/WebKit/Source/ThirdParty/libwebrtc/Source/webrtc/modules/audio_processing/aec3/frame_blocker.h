/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 23, 2021.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_FRAME_BLOCKER_H_
#define MODULES_AUDIO_PROCESSING_AEC3_FRAME_BLOCKER_H_

#include <stddef.h>

#include <vector>

#include "api/array_view.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/block.h"

namespace webrtc {

// Class for producing 64 sample multiband blocks from frames consisting of 2
// subframes of 80 samples.
class FrameBlocker {
 public:
  FrameBlocker(size_t num_bands, size_t num_channels);
  ~FrameBlocker();
  FrameBlocker(const FrameBlocker&) = delete;
  FrameBlocker& operator=(const FrameBlocker&) = delete;

  // Inserts one 80 sample multiband subframe from the multiband frame and
  // extracts one 64 sample multiband block.
  void InsertSubFrameAndExtractBlock(
      const std::vector<std::vector<rtc::ArrayView<float>>>& sub_frame,
      Block* block);
  // Reports whether a multiband block of 64 samples is available for
  // extraction.
  bool IsBlockAvailable() const;
  // Extracts a multiband block of 64 samples.
  void ExtractBlock(Block* block);

 private:
  const size_t num_bands_;
  const size_t num_channels_;
  std::vector<std::vector<std::vector<float>>> buffer_;
};
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_FRAME_BLOCKER_H_
