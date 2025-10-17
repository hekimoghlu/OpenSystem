/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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
#ifndef MODULES_RTP_RTCP_SOURCE_CAPTURE_CLOCK_OFFSET_UPDATER_H_
#define MODULES_RTP_RTCP_SOURCE_CAPTURE_CLOCK_OFFSET_UPDATER_H_

#include <stdint.h>

#include <optional>

#include "api/units/time_delta.h"

namespace webrtc {

//
// Helper class for calculating the clock offset against the capturer's clock.
//
// This is achieved by adjusting the estimated capture clock offset in received
// Absolute Capture Time RTP header extension (see
// https://webrtc.org/experiments/rtp-hdrext/abs-capture-time/), which
// represents the clock offset between a remote sender and the capturer, by
// adding local-to-remote clock offset.

class CaptureClockOffsetUpdater {
 public:
  // Adjusts remote_capture_clock_offset, which originates from Absolute Capture
  // Time RTP header extension, to get the local clock offset against the
  // capturer's clock.
  std::optional<int64_t> AdjustEstimatedCaptureClockOffset(
      std::optional<int64_t> remote_capture_clock_offset) const;

  // Sets the NTP clock offset between the sender system (which may be different
  // from the capture system) and the local system. This information is normally
  // provided by passing half the value of the Round-Trip Time estimation given
  // by RTCP sender reports (see DLSR/DLRR).
  //
  // Note that the value must be in Q32.32-formatted fixed-point seconds.
  void SetRemoteToLocalClockOffset(std::optional<int64_t> offset_q32x32);

  // Converts a signed Q32.32-formatted fixed-point to a TimeDelta.
  static std::optional<TimeDelta> ConvertsToTimeDela(
      std::optional<int64_t> q32x32);

 private:
  std::optional<int64_t> remote_to_local_clock_offset_;
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_CAPTURE_CLOCK_OFFSET_UPDATER_H_
