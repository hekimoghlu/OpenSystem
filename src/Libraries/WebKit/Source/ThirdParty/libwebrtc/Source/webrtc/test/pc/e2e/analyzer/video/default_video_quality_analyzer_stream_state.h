/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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
#ifndef TEST_PC_E2E_ANALYZER_VIDEO_DEFAULT_VIDEO_QUALITY_ANALYZER_STREAM_STATE_H_
#define TEST_PC_E2E_ANALYZER_VIDEO_DEFAULT_VIDEO_QUALITY_ANALYZER_STREAM_STATE_H_

#include <limits>
#include <optional>
#include <set>
#include <unordered_map>

#include "api/units/timestamp.h"
#include "system_wrappers/include/clock.h"
#include "test/pc/e2e/analyzer/video/dvqa/pausable_state.h"
#include "test/pc/e2e/analyzer/video/multi_reader_queue.h"

namespace webrtc {

// Represents a current state of video stream inside
// DefaultVideoQualityAnalyzer.
//
// Maintains the sequence of frames for each video stream and keeps track about
// which frames were seen by each of the possible stream receiver.
//
// Keeps information about which frames are alive and which are dead. Frame is
// alive if it contains VideoFrame payload for corresponding FrameInFlight
// object inside DefaultVideoQualityAnalyzer, otherwise frame is considered
// dead.
//
// Supports peer indexes from 0 to max(size_t) - 1.
class StreamState {
 public:
  StreamState(size_t sender,
              std::set<size_t> receivers,
              Timestamp stream_started_time,
              Clock* clock);

  size_t sender() const { return sender_; }
  Timestamp stream_started_time() const { return stream_started_time_; }

  void PushBack(uint16_t frame_id) { frame_ids_.PushBack(frame_id); }
  // Crash if state is empty.
  uint16_t PopFront(size_t peer);
  bool IsEmpty(size_t peer) const { return frame_ids_.IsEmpty(peer); }
  // Crash if state is empty.
  uint16_t Front(size_t peer) const { return frame_ids_.Front(peer).value(); }

  // Adds a new peer to the state. All currently alive frames will be expected
  // to be received by the newly added peer.
  void AddPeer(size_t peer);

  // Removes peer from the state. Frames that were expected to be received by
  // this peer will be removed from it. On the other hand last rendered frame
  // time for the removed peer will be preserved, because
  // DefaultVideoQualityAnalyzer still may request it for stats processing.
  void RemovePeer(size_t peer);

  // Returns a pointer to the PausableState of this stream for specified peer.
  // The pointer is owned by StreamState and guranteed to be not null.
  PausableState* GetPausableState(size_t peer);

  size_t GetAliveFramesCount() const {
    return frame_ids_.size(kAliveFramesQueueIndex);
  }

  void SetLastCapturedFrameTime(Timestamp time) {
    last_captured_frame_time_ = time;
  }
  std::optional<Timestamp> last_captured_frame_time() const {
    return last_captured_frame_time_;
  }

  void SetLastEncodedFrameTime(Timestamp time) {
    last_encoded_frame_time_ = time;
  }
  std::optional<Timestamp> last_encoded_frame_time() const {
    return last_encoded_frame_time_;
  }

  void SetLastRenderedFrameTime(size_t peer, Timestamp time);
  std::optional<Timestamp> last_rendered_frame_time(size_t peer) const;

 private:
  // Index of the `frame_ids_` queue which is used to track alive frames for
  // this stream.
  static constexpr size_t kAliveFramesQueueIndex =
      std::numeric_limits<size_t>::max();

  size_t GetLongestReceiverQueue() const;

  // Index of the owner. Owner's queue in `frame_ids_` will keep alive frames.
  const size_t sender_;
  const Timestamp stream_started_time_;
  Clock* const clock_;
  std::set<size_t> receivers_;
  // To correctly determine dropped frames we have to know sequence of frames
  // in each stream so we will keep a list of frame ids inside the stream.
  // This list is represented by multi head queue of frame ids with separate
  // head for each receiver. When the frame is rendered, we will pop ids from
  // the corresponding head until id will match with rendered one. All ids
  // before matched one can be considered as dropped:
  //
  // | frame_id1 |->| frame_id2 |->| frame_id3 |->| frame_id4 |
  //
  // If we received frame with id frame_id3, then we will pop frame_id1 and
  // frame_id2 and consider those frames as dropped and then compare received
  // frame with the one from `FrameInFlight` with id frame_id3.
  MultiReaderQueue<uint16_t> frame_ids_;
  std::optional<Timestamp> last_captured_frame_time_ = std::nullopt;
  std::optional<Timestamp> last_encoded_frame_time_ = std::nullopt;
  std::unordered_map<size_t, Timestamp> last_rendered_frame_time_;
  // Mapping from peer's index to pausable state for this receiver.
  std::unordered_map<size_t, PausableState> pausable_state_;
};

}  // namespace webrtc

#endif  // TEST_PC_E2E_ANALYZER_VIDEO_DEFAULT_VIDEO_QUALITY_ANALYZER_STREAM_STATE_H_
