/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 12, 2023.
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
#ifndef TEST_PC_E2E_ANALYZER_VIDEO_DVQA_FRAMES_STORAGE_H_
#define TEST_PC_E2E_ANALYZER_VIDEO_DVQA_FRAMES_STORAGE_H_

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "api/video/video_frame.h"
#include "system_wrappers/include/clock.h"

namespace webrtc {

// Stores video frames for DefaultVideoQualityAnalyzer. Frames are cleaned up
// when the time elapsed from their capture time exceeds `max_storage_duration`.
class FramesStorage {
 public:
  FramesStorage(TimeDelta max_storage_duration, Clock* clock)
      : max_storage_duration_(max_storage_duration), clock_(clock) {}
  FramesStorage(const FramesStorage&) = delete;
  FramesStorage& operator=(const FramesStorage&) = delete;
  FramesStorage(FramesStorage&&) = default;
  FramesStorage& operator=(FramesStorage&&) = default;

  // Adds frame to the storage. It is guaranteed to be stored at least
  // `max_storage_duration` from `captured_time`.
  //
  // Complexity: O(log(n))
  void Add(const VideoFrame& frame, Timestamp captured_time);

  // Complexity: O(1)
  std::optional<VideoFrame> Get(uint16_t frame_id);

  // Removes the frame identified by `frame_id` from the storage. No error
  // happens in case there isn't a frame identified by `frame_id`.
  //
  // Complexity: O(log(n))
  void Remove(uint16_t frame_id);

 private:
  struct HeapNode {
    VideoFrame frame;
    Timestamp captured_time;
  };

  void RemoveInternal(uint16_t frame_id);

  void Heapify(size_t index);
  void HeapifyUp(size_t index);
  void HeapifyDown(size_t index);

  // Complexity: O(#(of too old frames) * log(n))
  void RemoveTooOldFrames();

  TimeDelta max_storage_duration_;
  Clock* clock_;

  std::unordered_map<uint16_t, size_t> frame_id_index_;
  // Min-heap based on HeapNode::captured_time.
  std::vector<HeapNode> heap_;
};

}  // namespace webrtc

#endif  // TEST_PC_E2E_ANALYZER_VIDEO_DVQA_FRAMES_STORAGE_H_
