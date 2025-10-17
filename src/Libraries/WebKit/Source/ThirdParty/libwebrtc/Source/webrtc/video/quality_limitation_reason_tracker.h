/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 30, 2023.
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
#ifndef VIDEO_QUALITY_LIMITATION_REASON_TRACKER_H_
#define VIDEO_QUALITY_LIMITATION_REASON_TRACKER_H_

#include <map>

#include "common_video/include/quality_limitation_reason.h"
#include "system_wrappers/include/clock.h"

namespace webrtc {

// A tracker of quality limitation reasons. The quality limitation reason is the
// primary reason for limiting resolution and/or framerate (such as CPU or
// bandwidth limitations). The tracker keeps track of the current reason and the
// duration of time spent in each reason. See qualityLimitationReason[1],
// qualityLimitationDurations[2], and qualityLimitationResolutionChanges[3] in
// the webrtc-stats spec.
// Note that the specification defines the durations in seconds while the
// internal data structures defines it in milliseconds.
// [1]
// https://w3c.github.io/webrtc-stats/#dom-rtcoutboundrtpstreamstats-qualitylimitationreason
// [2]
// https://w3c.github.io/webrtc-stats/#dom-rtcoutboundrtpstreamstats-qualitylimitationdurations
// [3]
// https://w3c.github.io/webrtc-stats/#dom-rtcoutboundrtpstreamstats-qualitylimitationresolutionchanges
class QualityLimitationReasonTracker {
 public:
  // The caller is responsible for making sure `clock` outlives the tracker.
  explicit QualityLimitationReasonTracker(Clock* clock);

  // The current reason defaults to QualityLimitationReason::kNone.
  QualityLimitationReason current_reason() const;
  void SetReason(QualityLimitationReason reason);
  std::map<QualityLimitationReason, int64_t> DurationsMs() const;

 private:
  Clock* const clock_;
  QualityLimitationReason current_reason_;
  int64_t current_reason_updated_timestamp_ms_;
  // The total amount of time spent in each reason at time
  // `current_reason_updated_timestamp_ms_`. To get the total amount duration
  // so-far, including the time spent in `current_reason_` elapsed since the
  // last time `current_reason_` was updated, see DurationsMs().
  std::map<QualityLimitationReason, int64_t> durations_ms_;
};

}  // namespace webrtc

#endif  // VIDEO_QUALITY_LIMITATION_REASON_TRACKER_H_
