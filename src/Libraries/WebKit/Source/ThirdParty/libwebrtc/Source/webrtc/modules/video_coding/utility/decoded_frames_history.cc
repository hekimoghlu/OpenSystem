/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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
#include "modules/video_coding/utility/decoded_frames_history.h"

#include <algorithm>

#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {
namespace video_coding {

DecodedFramesHistory::DecodedFramesHistory(size_t window_size)
    : buffer_(window_size) {}

DecodedFramesHistory::~DecodedFramesHistory() = default;

void DecodedFramesHistory::InsertDecoded(int64_t frame_id, uint32_t timestamp) {
  last_decoded_frame_ = frame_id;
  last_decoded_frame_timestamp_ = timestamp;
  int new_index = FrameIdToIndex(frame_id);

  RTC_DCHECK(last_frame_id_ < frame_id);

  // Clears expired values from the cyclic buffer_.
  if (last_frame_id_) {
    int64_t id_jump = frame_id - *last_frame_id_;
    int last_index = FrameIdToIndex(*last_frame_id_);

    if (id_jump >= static_cast<int64_t>(buffer_.size())) {
      std::fill(buffer_.begin(), buffer_.end(), false);
    } else if (new_index > last_index) {
      std::fill(buffer_.begin() + last_index + 1, buffer_.begin() + new_index,
                false);
    } else {
      std::fill(buffer_.begin() + last_index + 1, buffer_.end(), false);
      std::fill(buffer_.begin(), buffer_.begin() + new_index, false);
    }
  }

  buffer_[new_index] = true;
  last_frame_id_ = frame_id;
}

bool DecodedFramesHistory::WasDecoded(int64_t frame_id) const {
  if (!last_frame_id_)
    return false;

  // Reference to the picture_id out of the stored should happen.
  if (frame_id <= *last_frame_id_ - static_cast<int64_t>(buffer_.size())) {
    RTC_LOG(LS_WARNING) << "Referencing a frame out of the window. "
                           "Assuming it was undecoded to avoid artifacts.";
    return false;
  }

  if (frame_id > last_frame_id_)
    return false;

  return buffer_[FrameIdToIndex(frame_id)];
}

void DecodedFramesHistory::Clear() {
  last_decoded_frame_timestamp_.reset();
  last_decoded_frame_.reset();
  std::fill(buffer_.begin(), buffer_.end(), false);
  last_frame_id_.reset();
}

std::optional<int64_t> DecodedFramesHistory::GetLastDecodedFrameId() const {
  return last_decoded_frame_;
}

std::optional<uint32_t> DecodedFramesHistory::GetLastDecodedFrameTimestamp()
    const {
  return last_decoded_frame_timestamp_;
}

int DecodedFramesHistory::FrameIdToIndex(int64_t frame_id) const {
  int m = frame_id % buffer_.size();
  return m >= 0 ? m : m + buffer_.size();
}

}  // namespace video_coding
}  // namespace webrtc
