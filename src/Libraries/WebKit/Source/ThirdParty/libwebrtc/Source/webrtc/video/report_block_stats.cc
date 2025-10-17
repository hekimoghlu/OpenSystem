/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 5, 2023.
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
#include "video/report_block_stats.h"

#include <algorithm>

namespace webrtc {

namespace {
int FractionLost(uint32_t num_lost_sequence_numbers,
                 uint32_t num_sequence_numbers) {
  if (num_sequence_numbers == 0) {
    return 0;
  }
  return ((num_lost_sequence_numbers * 255) + (num_sequence_numbers / 2)) /
         num_sequence_numbers;
}
}  // namespace

// Helper class for rtcp statistics.
ReportBlockStats::ReportBlockStats()
    : num_sequence_numbers_(0), num_lost_sequence_numbers_(0) {}

ReportBlockStats::~ReportBlockStats() {}

void ReportBlockStats::Store(uint32_t ssrc,
                             int packets_lost,
                             uint32_t extended_highest_sequence_number) {
  Report report;
  report.packets_lost = packets_lost;
  report.extended_highest_sequence_number = extended_highest_sequence_number;

  // Get diff with previous report block.
  const auto prev_report = prev_reports_.find(ssrc);
  if (prev_report != prev_reports_.end()) {
    int seq_num_diff = report.extended_highest_sequence_number -
                       prev_report->second.extended_highest_sequence_number;
    int cum_loss_diff = report.packets_lost - prev_report->second.packets_lost;
    if (seq_num_diff >= 0 && cum_loss_diff >= 0) {
      // Update total number of packets/lost packets.
      num_sequence_numbers_ += seq_num_diff;
      num_lost_sequence_numbers_ += cum_loss_diff;
    }
  }
  // Store current report block.
  prev_reports_[ssrc] = report;
}

int ReportBlockStats::FractionLostInPercent() const {
  if (num_sequence_numbers_ == 0) {
    return -1;
  }
  return FractionLost(num_lost_sequence_numbers_, num_sequence_numbers_) * 100 /
         255;
}

}  // namespace webrtc
