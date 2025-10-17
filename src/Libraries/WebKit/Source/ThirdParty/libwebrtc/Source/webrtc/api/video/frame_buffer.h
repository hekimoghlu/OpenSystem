/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 12, 2023.
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
#ifndef API_VIDEO_FRAME_BUFFER_H_
#define API_VIDEO_FRAME_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>

#include "absl/container/inlined_vector.h"
#include "api/field_trials_view.h"
#include "api/video/encoded_frame.h"
#include "modules/video_coding/utility/decoded_frames_history.h"

namespace webrtc {
// The high level idea of the FrameBuffer is to order frames received from the
// network into a decodable stream. Frames are order by frame ID, and grouped
// into temporal units by timestamp. A temporal unit is decodable after all
// referenced frames outside the unit has been decoded, and a temporal unit is
// continuous if all referenced frames are directly or indirectly decodable.
// The FrameBuffer is thread-unsafe.
class FrameBuffer {
 public:
  struct DecodabilityInfo {
    uint32_t next_rtp_timestamp;
    uint32_t last_rtp_timestamp;
  };

  // The `max_size` determines the maximum number of frames the buffer will
  // store, and max_decode_history determines how far back (by frame ID) the
  // buffer will store if a frame was decoded or not.
  FrameBuffer(int max_size,
              int max_decode_history,
              // TODO(hta): remove field trials!
              const FieldTrialsView& field_trials);
  FrameBuffer(const FrameBuffer&) = delete;
  FrameBuffer& operator=(const FrameBuffer&) = delete;
  ~FrameBuffer() = default;

  // Inserted frames may only reference backwards, and must have no duplicate
  // references. Frame insertion will fail if `frame` is a duplicate, has
  // already been decoded, invalid, or if the buffer is full and the frame is
  // not a keyframe. Returns true if the frame was successfully inserted.
  bool InsertFrame(std::unique_ptr<EncodedFrame> frame);

  // Mark all frames belonging to the next decodable temporal unit as decoded
  // and returns them.
  absl::InlinedVector<std::unique_ptr<EncodedFrame>, 4>
  ExtractNextDecodableTemporalUnit();

  // Drop all frames in the next decodable unit.
  void DropNextDecodableTemporalUnit();

  std::optional<int64_t> LastContinuousFrameId() const;
  std::optional<int64_t> LastContinuousTemporalUnitFrameId() const;
  std::optional<DecodabilityInfo> DecodableTemporalUnitsInfo() const;

  int GetTotalNumberOfContinuousTemporalUnits() const;
  int GetTotalNumberOfDroppedFrames() const;
  size_t CurrentSize() const;

 private:
  struct FrameInfo {
    std::unique_ptr<EncodedFrame> encoded_frame;
    bool continuous = false;
  };

  using FrameMap = std::map<int64_t, FrameInfo>;
  using FrameIterator = FrameMap::iterator;

  struct TemporalUnit {
    // Both first and last are inclusive.
    FrameIterator first_frame;
    FrameIterator last_frame;
  };

  bool IsContinuous(const FrameIterator& it) const;
  void PropagateContinuity(const FrameIterator& frame_it);
  void FindNextAndLastDecodableTemporalUnit();
  void Clear();

  const bool legacy_frame_id_jump_behavior_;
  const size_t max_size_;
  FrameMap frames_;
  std::optional<TemporalUnit> next_decodable_temporal_unit_;
  std::optional<DecodabilityInfo> decodable_temporal_units_info_;
  std::optional<int64_t> last_continuous_frame_id_;
  std::optional<int64_t> last_continuous_temporal_unit_frame_id_;
  video_coding::DecodedFramesHistory decoded_frame_history_;

  int num_continuous_temporal_units_ = 0;
  int num_dropped_frames_ = 0;
};

}  // namespace webrtc

#endif  // API_VIDEO_FRAME_BUFFER_H_
