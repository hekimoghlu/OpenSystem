/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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
#ifndef MODULES_RTP_RTCP_SOURCE_ABSOLUTE_CAPTURE_TIME_INTERPOLATOR_H_
#define MODULES_RTP_RTCP_SOURCE_ABSOLUTE_CAPTURE_TIME_INTERPOLATOR_H_

#include "api/array_view.h"
#include "api/rtp_headers.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/thread_annotations.h"
#include "system_wrappers/include/clock.h"

namespace webrtc {

//
// Helper class for interpolating the `AbsoluteCaptureTime` header extension.
//
// Supports the "timestamp interpolation" optimization:
//   A receiver SHOULD memorize the capture system (i.e. CSRC/SSRC), capture
//   timestamp, and RTP timestamp of the most recently received abs-capture-time
//   packet on each received stream. It can then use that information, in
//   combination with RTP timestamps of packets without abs-capture-time, to
//   extrapolate missing capture timestamps.
//
// See: https://webrtc.org/experiments/rtp-hdrext/abs-capture-time/
//
class AbsoluteCaptureTimeInterpolator {
 public:
  static constexpr TimeDelta kInterpolationMaxInterval = TimeDelta::Seconds(5);

  explicit AbsoluteCaptureTimeInterpolator(Clock* clock);

  // Returns the source (i.e. SSRC or CSRC) of the capture system.
  static uint32_t GetSource(uint32_t ssrc,
                            rtc::ArrayView<const uint32_t> csrcs);

  // Returns a received header extension, an interpolated header extension, or
  // `std::nullopt` if it's not possible to interpolate a header extension.
  std::optional<AbsoluteCaptureTime> OnReceivePacket(
      uint32_t source,
      uint32_t rtp_timestamp,
      int rtp_clock_frequency_hz,
      const std::optional<AbsoluteCaptureTime>& received_extension);

 private:
  friend class AbsoluteCaptureTimeSender;

  static uint64_t InterpolateAbsoluteCaptureTimestamp(
      uint32_t rtp_timestamp,
      int rtp_clock_frequency_hz,
      uint32_t last_rtp_timestamp,
      uint64_t last_absolute_capture_timestamp);

  bool ShouldInterpolateExtension(Timestamp receive_time,
                                  uint32_t source,
                                  uint32_t rtp_timestamp,
                                  int rtp_clock_frequency_hz) const
      RTC_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  Clock* const clock_;

  Mutex mutex_;

  // Time of the last received header extension eligible for interpolation,
  // MinusInfinity() if no extension was received, or last received one is
  // not eligible for interpolation.
  Timestamp last_receive_time_ RTC_GUARDED_BY(mutex_) =
      Timestamp::MinusInfinity();

  uint32_t last_source_ RTC_GUARDED_BY(mutex_);
  uint32_t last_rtp_timestamp_ RTC_GUARDED_BY(mutex_);
  int last_rtp_clock_frequency_hz_ RTC_GUARDED_BY(mutex_);
  AbsoluteCaptureTime last_received_extension_ RTC_GUARDED_BY(mutex_);
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_ABSOLUTE_CAPTURE_TIME_INTERPOLATOR_H_
