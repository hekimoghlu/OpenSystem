/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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
#include "api/test/create_peerconnection_quality_test_fixture.h"

#include <memory>
#include <string>
#include <utility>

#include "api/test/audio_quality_analyzer_interface.h"
#include "api/test/metrics/global_metrics_logger_and_exporter.h"
#include "api/test/peerconnection_quality_test_fixture.h"
#include "api/test/time_controller.h"
#include "api/test/video_quality_analyzer_interface.h"
#include "test/pc/e2e/peer_connection_quality_test.h"

namespace webrtc {
namespace webrtc_pc_e2e {

std::unique_ptr<PeerConnectionE2EQualityTestFixture>
CreatePeerConnectionE2EQualityTestFixture(
    std::string test_case_name,
    TimeController& time_controller,
    std::unique_ptr<AudioQualityAnalyzerInterface> audio_quality_analyzer,
    std::unique_ptr<VideoQualityAnalyzerInterface> video_quality_analyzer) {
  return std::make_unique<PeerConnectionE2EQualityTest>(
      std::move(test_case_name), time_controller,
      std::move(audio_quality_analyzer), std::move(video_quality_analyzer),
      test::GetGlobalMetricsLogger());
}

}  // namespace webrtc_pc_e2e
}  // namespace webrtc
