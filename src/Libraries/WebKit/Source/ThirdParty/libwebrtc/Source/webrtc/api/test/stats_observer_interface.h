/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 6, 2025.
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
#ifndef API_TEST_STATS_OBSERVER_INTERFACE_H_
#define API_TEST_STATS_OBSERVER_INTERFACE_H_

#include "absl/strings/string_view.h"
#include "api/scoped_refptr.h"
#include "api/stats/rtc_stats_report.h"

namespace webrtc {
namespace webrtc_pc_e2e {

// API is in development and can be changed without notice.
class StatsObserverInterface {
 public:
  virtual ~StatsObserverInterface() = default;

  // Method called when stats reports are available for the PeerConnection
  // identified by `pc_label`.
  virtual void OnStatsReports(
      absl::string_view pc_label,
      const rtc::scoped_refptr<const RTCStatsReport>& report) = 0;
};

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // API_TEST_STATS_OBSERVER_INTERFACE_H_
