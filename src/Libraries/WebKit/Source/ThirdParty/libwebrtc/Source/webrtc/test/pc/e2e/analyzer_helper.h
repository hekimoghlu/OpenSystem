/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 28, 2024.
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
#ifndef TEST_PC_E2E_ANALYZER_HELPER_H_
#define TEST_PC_E2E_ANALYZER_HELPER_H_

#include <map>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "api/sequence_checker.h"
#include "api/test/track_id_stream_info_map.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {
namespace webrtc_pc_e2e {

// This class is a utility that provides bookkeeping capabilities that
// are useful to associate stats reports track_ids to the remote stream info.
// The framework will populate an instance of this class and it will pass
// it to the Start method of Media Quality Analyzers.
// An instance of AnalyzerHelper must only be accessed from a single
// thread and since stats collection happens on the signaling thread,
// AddTrackToStreamMapping, GetStreamLabelFromTrackId and
// GetSyncGroupLabelFromTrackId must be invoked from the signaling thread. Get
// methods should be invoked only after all data is added. Mixing Get methods
// with adding new data may lead to undefined behavior.
class AnalyzerHelper : public TrackIdStreamInfoMap {
 public:
  AnalyzerHelper();

  void AddTrackToStreamMapping(absl::string_view track_id,
                               absl::string_view receiver_peer,
                               absl::string_view stream_label,
                               std::optional<std::string> sync_group);
  void AddTrackToStreamMapping(std::string track_id, std::string stream_label);
  void AddTrackToStreamMapping(std::string track_id,
                               std::string stream_label,
                               std::string sync_group);

  StreamInfo GetStreamInfoFromTrackId(
      absl::string_view track_id) const override;

 private:
  SequenceChecker signaling_sequence_checker_;
  std::map<std::string, StreamInfo> track_to_stream_map_
      RTC_GUARDED_BY(signaling_sequence_checker_);
};

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // TEST_PC_E2E_ANALYZER_HELPER_H_
