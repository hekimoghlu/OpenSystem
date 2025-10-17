/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 4, 2023.
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
#ifndef SYSTEM_WRAPPERS_INCLUDE_RTP_TO_NTP_ESTIMATOR_H_
#define SYSTEM_WRAPPERS_INCLUDE_RTP_TO_NTP_ESTIMATOR_H_

#include <stdint.h>

#include <list>
#include <optional>

#include "rtc_base/checks.h"
#include "rtc_base/numerics/sequence_number_unwrapper.h"
#include "system_wrappers/include/ntp_time.h"

namespace webrtc {

// Converts an RTP timestamp to the NTP domain.
// The class needs to be trained with (at least 2) RTP/NTP timestamp pairs from
// RTCP sender reports before the convertion can be done.
class RtpToNtpEstimator {
 public:
  static constexpr int kMaxInvalidSamples = 3;

  RtpToNtpEstimator() = default;
  RtpToNtpEstimator(const RtpToNtpEstimator&) = delete;
  RtpToNtpEstimator& operator=(const RtpToNtpEstimator&) = delete;
  ~RtpToNtpEstimator() = default;

  enum UpdateResult { kInvalidMeasurement, kSameMeasurement, kNewMeasurement };
  // Updates measurements with RTP/NTP timestamp pair from a RTCP sender report.
  UpdateResult UpdateMeasurements(NtpTime ntp, uint32_t rtp_timestamp);

  // Converts an RTP timestamp to the NTP domain.
  // Returns invalid NtpTime (i.e. NtpTime(0)) on failure.
  NtpTime Estimate(uint32_t rtp_timestamp) const;

  // Returns estimated rtp_timestamp frequency, or 0 on failure.
  double EstimatedFrequencyKhz() const;

 private:
  // Estimated parameters from RTP and NTP timestamp pairs in `measurements_`.
  // Defines linear estimation: NtpTime (in units of 1s/2^32) =
  //   `Parameters::slope` * rtp_timestamp + `Parameters::offset`.
  struct Parameters {
    double slope;
    double offset;
  };

  // RTP and NTP timestamp pair from a RTCP SR report.
  struct RtcpMeasurement {
    NtpTime ntp_time;
    int64_t unwrapped_rtp_timestamp;
  };

  void UpdateParameters();

  int consecutive_invalid_samples_ = 0;
  std::list<RtcpMeasurement> measurements_;
  std::optional<Parameters> params_;
  mutable RtpTimestampUnwrapper unwrapper_;
};
}  // namespace webrtc

#endif  // SYSTEM_WRAPPERS_INCLUDE_RTP_TO_NTP_ESTIMATOR_H_
