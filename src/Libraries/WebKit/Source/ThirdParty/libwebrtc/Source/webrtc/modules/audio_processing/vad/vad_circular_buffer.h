/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_VAD_VAD_CIRCULAR_BUFFER_H_
#define MODULES_AUDIO_PROCESSING_VAD_VAD_CIRCULAR_BUFFER_H_

#include <memory>

namespace webrtc {

// A circular buffer tailored to the need of this project. It stores last
// K samples of the input, and keeps track of the mean of the last samples.
//
// It is used in class "PitchBasedActivity" to keep track of posterior
// probabilities in the past few seconds. The posterior probabilities are used
// to recursively update prior probabilities.
class VadCircularBuffer {
 public:
  static VadCircularBuffer* Create(int buffer_size);
  ~VadCircularBuffer();

  // If buffer is wrapped around.
  bool is_full() const { return is_full_; }
  // Get the oldest entry in the buffer.
  double Oldest() const;
  // Insert new value into the buffer.
  void Insert(double value);
  // Reset buffer, forget the past, start fresh.
  void Reset();

  // The mean value of the elements in the buffer. The return value is zero if
  // buffer is empty, i.e. no value is inserted.
  double Mean();
  // Remove transients. If the values exceed `val_threshold` for a period
  // shorter then or equal to `width_threshold`, then that period is considered
  // transient and set to zero.
  int RemoveTransient(int width_threshold, double val_threshold);

 private:
  explicit VadCircularBuffer(int buffer_size);
  // Get previous values. |index = 0| corresponds to the most recent
  // insertion. |index = 1| is the one before the most recent insertion, and
  // so on.
  int Get(int index, double* value) const;
  // Set a given position to `value`. `index` is interpreted as above.
  int Set(int index, double value);
  // Return the number of valid elements in the buffer.
  int BufferLevel();

  // Convert an index with the interpretation as get() method to the
  // corresponding linear index.
  int ConvertToLinearIndex(int* index) const;

  std::unique_ptr<double[]> buffer_;
  bool is_full_;
  int index_;
  int buffer_size_;
  double sum_;
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_PROCESSING_VAD_VAD_CIRCULAR_BUFFER_H_
