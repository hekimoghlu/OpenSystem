/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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
#ifndef MODULES_DESKTOP_CAPTURE_SCREEN_CAPTURE_FRAME_QUEUE_H_
#define MODULES_DESKTOP_CAPTURE_SCREEN_CAPTURE_FRAME_QUEUE_H_

#include <memory>

namespace webrtc {

// Represents a queue of reusable video frames. Provides access to the 'current'
// frame - the frame that the caller is working with at the moment, and to the
// 'previous' frame - the predecessor of the current frame swapped by
// MoveToNextFrame() call, if any.
//
// The caller is expected to (re)allocate frames if current_frame() returns
// NULL. The caller can mark all frames in the queue for reallocation (when,
// say, frame dimensions change). The queue records which frames need updating
// which the caller can query.
//
// Frame consumer is expected to never hold more than kQueueLength frames
// created by this function and it should release the earliest one before trying
// to capture a new frame (i.e. before MoveToNextFrame() is called).
template <typename FrameType>
class ScreenCaptureFrameQueue {
 public:
  ScreenCaptureFrameQueue() = default;
  ~ScreenCaptureFrameQueue() = default;

  ScreenCaptureFrameQueue(const ScreenCaptureFrameQueue&) = delete;
  ScreenCaptureFrameQueue& operator=(const ScreenCaptureFrameQueue&) = delete;

  // Moves to the next frame in the queue, moving the 'current' frame to become
  // the 'previous' one.
  void MoveToNextFrame() { current_ = (current_ + 1) % kQueueLength; }

  // Replaces the current frame with a new one allocated by the caller. The
  // existing frame (if any) is destroyed. Takes ownership of `frame`.
  void ReplaceCurrentFrame(std::unique_ptr<FrameType> frame) {
    frames_[current_] = std::move(frame);
  }

  // Marks all frames obsolete and resets the previous frame pointer. No
  // frames are freed though as the caller can still access them.
  void Reset() {
    for (int i = 0; i < kQueueLength; i++) {
      frames_[i].reset();
    }
    current_ = 0;
  }

  FrameType* current_frame() const { return frames_[current_].get(); }

  FrameType* previous_frame() const {
    return frames_[(current_ + kQueueLength - 1) % kQueueLength].get();
  }

 private:
  // Index of the current frame.
  int current_ = 0;

  static const int kQueueLength = 2;
  std::unique_ptr<FrameType> frames_[kQueueLength];
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_SCREEN_CAPTURE_FRAME_QUEUE_H_
