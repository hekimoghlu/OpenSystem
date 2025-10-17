/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC_LOUDNESS_HISTOGRAM_H_
#define MODULES_AUDIO_PROCESSING_AGC_LOUDNESS_HISTOGRAM_H_

#include <stdint.h>

#include <memory>

namespace webrtc {

// This class implements the histogram of loudness with circular buffers so that
// the histogram tracks the last T seconds of the loudness.
class LoudnessHistogram {
 public:
  // Create a non-sliding LoudnessHistogram.
  static LoudnessHistogram* Create();

  // Create a sliding LoudnessHistogram, i.e. the histogram represents the last
  // `window_size` samples.
  static LoudnessHistogram* Create(int window_size);
  ~LoudnessHistogram();

  // Insert RMS and the corresponding activity probability.
  void Update(double rms, double activity_probability);

  // Reset the histogram, forget the past.
  void Reset();

  // Current loudness, which is actually the mean of histogram in loudness
  // domain.
  double CurrentRms() const;

  // Sum of the histogram content.
  double AudioContent() const;

  // Number of times the histogram has been updated.
  int num_updates() const { return num_updates_; }

 private:
  LoudnessHistogram();
  explicit LoudnessHistogram(int window);

  // Find the histogram bin associated with the given `rms`.
  int GetBinIndex(double rms);

  void RemoveOldestEntryAndUpdate();
  void InsertNewestEntryAndUpdate(int activity_prob_q10, int hist_index);
  void UpdateHist(int activity_prob_q10, int hist_index);
  void RemoveTransient();

  // Number of histogram bins.
  static const int kHistSize = 77;

  // Number of times the histogram is updated
  int num_updates_;
  // Audio content, this should be equal to the sum of the components of
  // `bin_count_q10_`.
  int64_t audio_content_q10_;

  // LoudnessHistogram of input RMS in Q10 with `kHistSize_` bins. In each
  // 'Update(),' we increment the associated histogram-bin with the given
  // probability. The increment is implemented in Q10 to avoid rounding errors.
  int64_t bin_count_q10_[kHistSize];

  // Circular buffer for probabilities
  std::unique_ptr<int[]> activity_probability_;
  // Circular buffer for histogram-indices of probabilities.
  std::unique_ptr<int[]> hist_bin_index_;
  // Current index of circular buffer, where the newest data will be written to,
  // therefore, pointing to the oldest data if buffer is full.
  int buffer_index_;
  // Indicating if buffer is full and we had a wrap around.
  int buffer_is_full_;
  // Size of circular buffer.
  int len_circular_buffer_;
  int len_high_activity_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC_LOUDNESS_HISTOGRAM_H_
