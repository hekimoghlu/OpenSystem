/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_CLIPPING_PREDICTOR_LEVEL_BUFFER_H_
#define MODULES_AUDIO_PROCESSING_AGC2_CLIPPING_PREDICTOR_LEVEL_BUFFER_H_

#include <memory>
#include <optional>
#include <vector>

namespace webrtc {

// A circular buffer to store frame-wise `Level` items for clipping prediction.
// The current implementation is not optimized for large buffer lengths.
class ClippingPredictorLevelBuffer {
 public:
  struct Level {
    float average;
    float max;
    bool operator==(const Level& level) const;
  };

  // Recommended maximum capacity. It is possible to create a buffer with a
  // larger capacity, but the implementation is not optimized for large values.
  static constexpr int kMaxCapacity = 100;

  // Ctor. Sets the buffer capacity to max(1, `capacity`) and logs a warning
  // message if the capacity is greater than `kMaxCapacity`.
  explicit ClippingPredictorLevelBuffer(int capacity);
  ~ClippingPredictorLevelBuffer() {}
  ClippingPredictorLevelBuffer(const ClippingPredictorLevelBuffer&) = delete;
  ClippingPredictorLevelBuffer& operator=(const ClippingPredictorLevelBuffer&) =
      delete;

  void Reset();

  // Returns the current number of items stored in the buffer.
  int Size() const { return size_; }

  // Returns the capacity of the buffer.
  int Capacity() const { return data_.size(); }

  // Adds a `level` item into the circular buffer `data_`. Stores at most
  // `Capacity()` items. If more items are pushed, the new item replaces the
  // least recently pushed item.
  void Push(Level level);

  // If at least `num_items` + `delay` items have been pushed, returns the
  // average and maximum value for the `num_items` most recently pushed items
  // from `delay` to `delay` - `num_items` (a delay equal to zero corresponds
  // to the most recently pushed item). The value of `delay` is limited to
  // [0, N] and `num_items` to [1, M] where N + M is the capacity of the buffer.
  std::optional<Level> ComputePartialMetrics(int delay, int num_items) const;

 private:
  int tail_;
  int size_;
  std::vector<Level> data_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_CLIPPING_PREDICTOR_LEVEL_BUFFER_H_
