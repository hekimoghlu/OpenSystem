/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 10, 2023.
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
#include "modules/rtp_rtcp/source/capture_clock_offset_updater.h"

#include "system_wrappers/include/ntp_time.h"

namespace webrtc {

std::optional<int64_t>
CaptureClockOffsetUpdater::AdjustEstimatedCaptureClockOffset(
    std::optional<int64_t> remote_capture_clock_offset) const {
  if (remote_capture_clock_offset == std::nullopt ||
      remote_to_local_clock_offset_ == std::nullopt) {
    return std::nullopt;
  }

  // Do calculations as "unsigned" to make overflows deterministic.
  return static_cast<uint64_t>(*remote_capture_clock_offset) +
         static_cast<uint64_t>(*remote_to_local_clock_offset_);
}

std::optional<TimeDelta> CaptureClockOffsetUpdater::ConvertsToTimeDela(
    std::optional<int64_t> q32x32) {
  if (q32x32 == std::nullopt) {
    return std::nullopt;
  }
  return TimeDelta::Millis(Q32x32ToInt64Ms(*q32x32));
}

void CaptureClockOffsetUpdater::SetRemoteToLocalClockOffset(
    std::optional<int64_t> offset_q32x32) {
  remote_to_local_clock_offset_ = offset_q32x32;
}

}  // namespace webrtc
