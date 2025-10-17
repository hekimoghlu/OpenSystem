/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 26, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_BLOCK_FRAMER_H_
#define MODULES_AUDIO_PROCESSING_AEC3_BLOCK_FRAMER_H_

#include <vector>

#include "api/array_view.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/block.h"

namespace webrtc {

// Class for producing frames consisting of 2 subframes of 80 samples each
// from 64 sample blocks. The class is designed to work together with the
// FrameBlocker class which performs the reverse conversion. Used together with
// that, this class produces output frames are the same rate as frames are
// received by the FrameBlocker class. Note that the internal buffers will
// overrun if any other rate of packets insertion is used.
class BlockFramer {
 public:
  BlockFramer(size_t num_bands, size_t num_channels);
  ~BlockFramer();
  BlockFramer(const BlockFramer&) = delete;
  BlockFramer& operator=(const BlockFramer&) = delete;

  // Adds a 64 sample block into the data that will form the next output frame.
  void InsertBlock(const Block& block);
  // Adds a 64 sample block and extracts an 80 sample subframe.
  void InsertBlockAndExtractSubFrame(
      const Block& block,
      std::vector<std::vector<rtc::ArrayView<float>>>* sub_frame);

 private:
  const size_t num_bands_;
  const size_t num_channels_;
  std::vector<std::vector<std::vector<float>>> buffer_;
};
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_BLOCK_FRAMER_H_
