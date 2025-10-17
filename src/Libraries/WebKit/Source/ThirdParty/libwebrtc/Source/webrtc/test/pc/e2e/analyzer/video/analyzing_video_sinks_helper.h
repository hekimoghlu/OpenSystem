/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 15, 2024.
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
#ifndef TEST_PC_E2E_ANALYZER_VIDEO_ANALYZING_VIDEO_SINKS_HELPER_H_
#define TEST_PC_E2E_ANALYZER_VIDEO_ANALYZING_VIDEO_SINKS_HELPER_H_

#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "api/test/pclf/media_configuration.h"
#include "api/test/video/video_frame_writer.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {
namespace webrtc_pc_e2e {

// Registry of known video configs and video writers.
// This class is thread safe.
class AnalyzingVideoSinksHelper {
 public:
  // Adds config in the registry. If config with such stream label was
  // registered before, the new value will override the old one.
  void AddConfig(absl::string_view sender_peer_name, VideoConfig config);
  std::optional<std::pair<std::string, VideoConfig>> GetPeerAndConfig(
      absl::string_view stream_label);
  // Removes video config for specified stream label. If there are no know video
  // config for such stream label - does nothing.
  void RemoveConfig(absl::string_view stream_label);

  // Takes ownership of the provided video writer. All video writers owned by
  // this class will be closed during `AnalyzingVideoSinksHelper` destruction
  // and guaranteed to be alive either until explicitly removed by
  // `CloseAndRemoveVideoWriters` or until `AnalyzingVideoSinksHelper` is
  // destroyed.
  //
  // Returns pointer to the added writer. Ownership is maintained by
  // `AnalyzingVideoSinksHelper`.
  test::VideoFrameWriter* AddVideoWriter(
      std::unique_ptr<test::VideoFrameWriter> video_writer);
  // For each provided `writers_to_close`, if it is known, will close and
  // destroy it, otherwise does nothing with it.
  void CloseAndRemoveVideoWriters(
      std::set<test::VideoFrameWriter*> writers_to_close);

  // Removes all added configs and close and removes all added writers.
  void Clear();

 private:
  Mutex mutex_;
  std::map<std::string, std::pair<std::string, VideoConfig>> video_configs_
      RTC_GUARDED_BY(mutex_);
  std::list<std::unique_ptr<test::VideoFrameWriter>> video_writers_
      RTC_GUARDED_BY(mutex_);
};

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // TEST_PC_E2E_ANALYZER_VIDEO_ANALYZING_VIDEO_SINKS_HELPER_H_
