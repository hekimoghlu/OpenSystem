/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 20, 2025.
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
#include "modules/rtp_rtcp/include/report_block_data.h"

#include "rtc_base/checks.h"

namespace webrtc {

TimeDelta ReportBlockData::jitter(int rtp_clock_rate_hz) const {
  RTC_DCHECK_GT(rtp_clock_rate_hz, 0);
  // Conversion to TimeDelta and division are swapped to avoid conversion
  // to/from floating point types.
  return TimeDelta::Seconds(jitter()) / rtp_clock_rate_hz;
}

// TODO: bugs.webrtc.org/370535296 - Remove the utc timestamp when linked
// issue is fixed.
void ReportBlockData::SetReportBlock(uint32_t sender_ssrc,
                                     const rtcp::ReportBlock& report_block,
                                     Timestamp report_block_timestamp_utc,
                                     Timestamp report_block_timestamp) {
  sender_ssrc_ = sender_ssrc;
  source_ssrc_ = report_block.source_ssrc();
  fraction_lost_raw_ = report_block.fraction_lost();
  cumulative_lost_ = report_block.cumulative_lost();
  extended_highest_sequence_number_ = report_block.extended_high_seq_num();
  jitter_ = report_block.jitter();
  report_block_timestamp_utc_ = report_block_timestamp_utc;
  report_block_timestamp_ = report_block_timestamp;
}

void ReportBlockData::AddRoundTripTimeSample(TimeDelta rtt) {
  last_rtt_ = rtt;
  sum_rtt_ += rtt;
  ++num_rtts_;
}

}  // namespace webrtc
