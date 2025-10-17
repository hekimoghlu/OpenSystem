/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 22, 2022.
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
#include "test/pc/e2e/analyzer_helper.h"

#include <string>
#include <utility>

namespace webrtc {
namespace webrtc_pc_e2e {

AnalyzerHelper::AnalyzerHelper() {
  signaling_sequence_checker_.Detach();
}

void AnalyzerHelper::AddTrackToStreamMapping(
    absl::string_view track_id,
    absl::string_view receiver_peer,
    absl::string_view stream_label,
    std::optional<std::string> sync_group) {
  RTC_DCHECK_RUN_ON(&signaling_sequence_checker_);
  track_to_stream_map_.insert(
      {std::string(track_id),
       StreamInfo{.receiver_peer = std::string(receiver_peer),
                  .stream_label = std::string(stream_label),
                  .sync_group = sync_group.has_value()
                                    ? *sync_group
                                    : std::string(stream_label)}});
}

void AnalyzerHelper::AddTrackToStreamMapping(std::string track_id,
                                             std::string stream_label) {
  RTC_DCHECK_RUN_ON(&signaling_sequence_checker_);
  track_to_stream_map_.insert(
      {std::move(track_id), StreamInfo{stream_label, stream_label}});
}

void AnalyzerHelper::AddTrackToStreamMapping(std::string track_id,
                                             std::string stream_label,
                                             std::string sync_group) {
  RTC_DCHECK_RUN_ON(&signaling_sequence_checker_);
  track_to_stream_map_.insert(
      {std::move(track_id),
       StreamInfo{std::move(stream_label), std::move(sync_group)}});
}

AnalyzerHelper::StreamInfo AnalyzerHelper::GetStreamInfoFromTrackId(
    absl::string_view track_id) const {
  RTC_DCHECK_RUN_ON(&signaling_sequence_checker_);
  auto track_to_stream_pair = track_to_stream_map_.find(std::string(track_id));
  RTC_CHECK(track_to_stream_pair != track_to_stream_map_.end());
  return track_to_stream_pair->second;
}

}  // namespace webrtc_pc_e2e
}  // namespace webrtc
