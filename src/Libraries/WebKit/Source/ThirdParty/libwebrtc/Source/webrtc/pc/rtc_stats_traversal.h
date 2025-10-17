/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 22, 2023.
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
#ifndef PC_RTC_STATS_TRAVERSAL_H_
#define PC_RTC_STATS_TRAVERSAL_H_

#include <string>
#include <vector>

#include "api/scoped_refptr.h"
#include "api/stats/rtc_stats.h"
#include "api/stats/rtc_stats_report.h"

namespace webrtc {

// Traverses the stats graph, taking all stats objects that are directly or
// indirectly accessible from and including the stats objects identified by
// `ids`, returning them as a new stats report.
// This is meant to be used to implement the stats selection algorithm.
// https://w3c.github.io/webrtc-pc/#dfn-stats-selection-algorithm
rtc::scoped_refptr<RTCStatsReport> TakeReferencedStats(
    rtc::scoped_refptr<RTCStatsReport> report,
    const std::vector<std::string>& ids);

// Gets pointers to the string values of any members in `stats` that are used as
// references for looking up other stats objects in the same report by ID. The
// pointers are valid for the lifetime of `stats` assumings its members are not
// modified.
//
// For example, RTCCodecStats contains "transportId"
// (RTCCodecStats::transport_id) referencing an RTCTransportStats.
// https://w3c.github.io/webrtc-stats/#dom-rtccodecstats-transportid
std::vector<const std::string*> GetStatsReferencedIds(const RTCStats& stats);

}  // namespace webrtc

#endif  // PC_RTC_STATS_TRAVERSAL_H_
