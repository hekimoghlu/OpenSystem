/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 15, 2024.
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
#ifndef MODULES_VIDEO_CODING_UTILITY_DECODED_FRAMES_HISTORY_H_
#define MODULES_VIDEO_CODING_UTILITY_DECODED_FRAMES_HISTORY_H_

#include <stdint.h>

#include <bitset>
#include <optional>
#include <vector>

#include "api/video/encoded_frame.h"

namespace webrtc {
namespace video_coding {

class DecodedFramesHistory {
 public:
  // window_size - how much frames back to the past are actually remembered.
  explicit DecodedFramesHistory(size_t window_size);
  ~DecodedFramesHistory();
  // Called for each decoded frame. Assumes frame id's are non-decreasing.
  void InsertDecoded(int64_t frame_id, uint32_t timestamp);
  // Query if the following (frame_id, spatial_id) pair was inserted before.
  // Should be at most less by window_size-1 than the last inserted frame id.
  bool WasDecoded(int64_t frame_id) const;

  void Clear();

  std::optional<int64_t> GetLastDecodedFrameId() const;
  std::optional<uint32_t> GetLastDecodedFrameTimestamp() const;

 private:
  int FrameIdToIndex(int64_t frame_id) const;

  std::vector<bool> buffer_;
  std::optional<int64_t> last_frame_id_;
  std::optional<int64_t> last_decoded_frame_;
  std::optional<uint32_t> last_decoded_frame_timestamp_;
};

}  // namespace video_coding
}  // namespace webrtc
#endif  // MODULES_VIDEO_CODING_UTILITY_DECODED_FRAMES_HISTORY_H_
