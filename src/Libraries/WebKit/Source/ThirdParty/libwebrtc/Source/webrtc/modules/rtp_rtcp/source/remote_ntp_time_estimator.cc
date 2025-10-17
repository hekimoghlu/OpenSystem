/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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
#include "modules/rtp_rtcp/include/remote_ntp_time_estimator.h"

#include <cstdint>

#include "modules/rtp_rtcp/source/ntp_time_util.h"
#include "rtc_base/logging.h"
#include "system_wrappers/include/clock.h"
#include "system_wrappers/include/ntp_time.h"

namespace webrtc {

namespace {

constexpr int kMinimumNumberOfSamples = 2;
constexpr TimeDelta kTimingLogInterval = TimeDelta::Seconds(10);
constexpr int kClocksOffsetSmoothingWindow = 100;

// Subtracts two NtpTime values keeping maximum precision.
int64_t Subtract(NtpTime minuend, NtpTime subtrahend) {
  uint64_t a = static_cast<uint64_t>(minuend);
  uint64_t b = static_cast<uint64_t>(subtrahend);
  return a >= b ? static_cast<int64_t>(a - b) : -static_cast<int64_t>(b - a);
}

NtpTime Add(NtpTime lhs, int64_t rhs) {
  uint64_t result = static_cast<uint64_t>(lhs);
  if (rhs >= 0) {
    result += static_cast<uint64_t>(rhs);
  } else {
    result -= static_cast<uint64_t>(-rhs);
  }
  return NtpTime(result);
}

}  // namespace

// TODO(wu): Refactor this class so that it can be shared with
// vie_sync_module.cc.
RemoteNtpTimeEstimator::RemoteNtpTimeEstimator(Clock* clock)
    : clock_(clock),
      ntp_clocks_offset_estimator_(kClocksOffsetSmoothingWindow) {}

bool RemoteNtpTimeEstimator::UpdateRtcpTimestamp(TimeDelta rtt,
                                                 NtpTime sender_send_time,
                                                 uint32_t rtp_timestamp) {
  switch (rtp_to_ntp_.UpdateMeasurements(sender_send_time, rtp_timestamp)) {
    case RtpToNtpEstimator::kInvalidMeasurement:
      return false;
    case RtpToNtpEstimator::kSameMeasurement:
      // No new RTCP SR since last time this function was called.
      return true;
    case RtpToNtpEstimator::kNewMeasurement:
      break;
  }

  // Assume connection is symmetric and thus time to deliver the packet is half
  // the round trip time.
  int64_t deliver_time_ntp = ToNtpUnits(rtt) / 2;

  // Update extrapolator with the new arrival time.
  NtpTime receiver_arrival_time = clock_->CurrentNtpTime();
  int64_t remote_to_local_clocks_offset =
      Subtract(receiver_arrival_time, sender_send_time) - deliver_time_ntp;
  ntp_clocks_offset_estimator_.Insert(remote_to_local_clocks_offset);
  return true;
}

NtpTime RemoteNtpTimeEstimator::EstimateNtp(uint32_t rtp_timestamp) {
  NtpTime sender_capture = rtp_to_ntp_.Estimate(rtp_timestamp);
  if (!sender_capture.Valid()) {
    return sender_capture;
  }

  int64_t remote_to_local_clocks_offset =
      ntp_clocks_offset_estimator_.GetFilteredValue();
  NtpTime receiver_capture = Add(sender_capture, remote_to_local_clocks_offset);

  Timestamp now = clock_->CurrentTime();
  if (now - last_timing_log_ > kTimingLogInterval) {
    RTC_LOG(LS_INFO) << "RTP timestamp: " << rtp_timestamp
                     << " in NTP clock: " << sender_capture.ToMs()
                     << " estimated time in receiver NTP clock: "
                     << receiver_capture.ToMs();
    last_timing_log_ = now;
  }

  return receiver_capture;
}

std::optional<int64_t>
RemoteNtpTimeEstimator::EstimateRemoteToLocalClockOffset() {
  if (ntp_clocks_offset_estimator_.GetNumberOfSamplesStored() <
      kMinimumNumberOfSamples) {
    return std::nullopt;
  }
  return ntp_clocks_offset_estimator_.GetFilteredValue();
}

}  // namespace webrtc
