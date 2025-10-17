/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 13, 2022.
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
#ifndef COMMON_VIDEO_FRAME_RATE_ESTIMATOR_H_
#define COMMON_VIDEO_FRAME_RATE_ESTIMATOR_H_

#include <deque>
#include <optional>

#include "api/units/time_delta.h"
#include "api/units/timestamp.h"

namespace webrtc {

// Class used to estimate a frame-rate using inter-frame intervals.
// Some notes on usage:
// This class is intended to accurately estimate the frame rate during a
// continuous stream. Unlike a traditional rate estimator that looks at number
// of data points within a time window, if the input stops this implementation
// will not smoothly fall down towards 0. This is done so that the estimated
// fps is not affected by edge conditions like if we sample just before or just
// after the next frame.
// To avoid problems if a stream is stopped and restarted (where estimated fps
// could look too low), users of this class should explicitly call Reset() on
// restart.
// Also note that this class is not thread safe, it's up to the user to guard
// against concurrent access.
class FrameRateEstimator {
 public:
  explicit FrameRateEstimator(TimeDelta averaging_window);

  // Insert a frame, potentially culling old frames that falls outside the
  // averaging window.
  void OnFrame(Timestamp time);

  // Get the current average FPS, based on the frames currently in the window.
  std::optional<double> GetAverageFps() const;

  // Move the window so it ends at `now`, and return the new fps estimate.
  std::optional<double> GetAverageFps(Timestamp now);

  // Completely clear the averaging window.
  void Reset();

 private:
  void CullOld(Timestamp now);
  const TimeDelta averaging_window_;
  std::deque<Timestamp> frame_times_;
};

}  // namespace webrtc

#endif  // COMMON_VIDEO_FRAME_RATE_ESTIMATOR_H_
