/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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
#ifndef API_TEST_AUDIO_QUALITY_ANALYZER_INTERFACE_H_
#define API_TEST_AUDIO_QUALITY_ANALYZER_INTERFACE_H_

#include <string>

#include "api/test/stats_observer_interface.h"
#include "api/test/track_id_stream_info_map.h"

namespace webrtc {
namespace webrtc_pc_e2e {

// API is in development. Can be changed/removed without notice.
class AudioQualityAnalyzerInterface : public StatsObserverInterface {
 public:
  ~AudioQualityAnalyzerInterface() override = default;

  // Will be called by the framework before the test.
  // `test_case_name` is name of test case, that should be used to report all
  // audio metrics.
  // `analyzer_helper` is a pointer to a class that will allow track_id to
  // stream_id matching. The caller is responsible for ensuring the
  // AnalyzerHelper outlives the instance of the AudioQualityAnalyzerInterface.
  virtual void Start(std::string test_case_name,
                     TrackIdStreamInfoMap* analyzer_helper) = 0;

  // Will be called by the framework at the end of the test. The analyzer
  // has to finalize all its stats and it should report them.
  virtual void Stop() = 0;
};

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // API_TEST_AUDIO_QUALITY_ANALYZER_INTERFACE_H_
