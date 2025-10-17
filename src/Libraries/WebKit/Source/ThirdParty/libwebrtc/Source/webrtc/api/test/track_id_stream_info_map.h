/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 4, 2024.
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
#ifndef API_TEST_TRACK_ID_STREAM_INFO_MAP_H_
#define API_TEST_TRACK_ID_STREAM_INFO_MAP_H_

#include <string>

#include "absl/strings/string_view.h"

namespace webrtc {
namespace webrtc_pc_e2e {

// Instances of `TrackIdStreamInfoMap` provide bookkeeping capabilities that
// are useful to associate stats reports track_ids to the remote stream info.
class TrackIdStreamInfoMap {
 public:
  struct StreamInfo {
    std::string receiver_peer;
    std::string stream_label;
    std::string sync_group;
  };

  virtual ~TrackIdStreamInfoMap() = default;

  // These methods must be called on the same thread where
  // StatsObserverInterface::OnStatsReports is invoked.

  // Precondition: `track_id` must be already mapped to stream info.
  virtual StreamInfo GetStreamInfoFromTrackId(
      absl::string_view track_id) const = 0;
};

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // API_TEST_TRACK_ID_STREAM_INFO_MAP_H_
