/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 12, 2022.
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
#ifndef VIDEO_REPORT_BLOCK_STATS_H_
#define VIDEO_REPORT_BLOCK_STATS_H_

#include <stdint.h>

#include <map>

namespace webrtc {

// TODO(nisse): Usefulness of this class is somewhat unclear. The inputs are
// cumulative counters, from which we compute deltas, and then accumulate the
// deltas. May be needed on the send side, to handle wraparound in the short
// counters received over RTCP, but should not be needed on the receive side
// where we can use large enough types for all counters we need.

// Helper class for rtcp statistics.
class ReportBlockStats {
 public:
  ReportBlockStats();
  ~ReportBlockStats();

  // Updates stats and stores report block.
  void Store(uint32_t ssrc,
             int packets_lost,
             uint32_t extended_highest_sequence_number);

  // Returns the total fraction of lost packets (or -1 if less than two report
  // blocks have been stored).
  int FractionLostInPercent() const;

 private:
  // The information from an RTCP report block that we need.
  struct Report {
    uint32_t extended_highest_sequence_number;
    int32_t packets_lost;
  };

  // The total number of packets/lost packets.
  uint32_t num_sequence_numbers_;
  uint32_t num_lost_sequence_numbers_;

  // Map holding the last stored report (mapped by the source SSRC).
  std::map<uint32_t, Report> prev_reports_;
};

}  // namespace webrtc

#endif  // VIDEO_REPORT_BLOCK_STATS_H_
