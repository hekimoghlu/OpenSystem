/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 4, 2025.
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
#include "test/gtest.h"
#include "test/peer_scenario/peer_scenario.h"
#include "test/peer_scenario/peer_scenario_client.h"

namespace webrtc {
namespace test {
#if defined(WEBRTC_WIN)
#define MAYBE_PsnrIsCollected DISABLED_PsnrIsCollected
#else
#define MAYBE_PsnrIsCollected PsnrIsCollected
#endif
TEST(PeerScenarioQualityTest, MAYBE_PsnrIsCollected) {
  VideoQualityAnalyzer analyzer;
  {
    PeerScenario s(*test_info_);
    auto caller = s.CreateClient(PeerScenarioClient::Config());
    auto callee = s.CreateClient(PeerScenarioClient::Config());
    PeerScenarioClient::VideoSendTrackConfig video_conf;
    video_conf.generator.squares_video->framerate = 20;
    auto video = caller->CreateVideo("VIDEO", video_conf);
    auto link_builder = s.net()->NodeBuilder().delay_ms(100).capacity_kbps(600);
    s.AttachVideoQualityAnalyzer(&analyzer, video.track.get(), callee);
    s.SimpleConnection(caller, callee, {link_builder.Build().node},
                       {link_builder.Build().node});
    s.ProcessMessages(TimeDelta::Seconds(2));
    // Exit scope to ensure that there's no pending tasks reporting to analyzer.
  }

  // We expect ca 40 frames to be produced, but to avoid flakiness on slow
  // machines we only test for 10.
  EXPECT_GT(analyzer.stats().render.count, 10);
  EXPECT_GT(analyzer.stats().psnr_with_freeze.Mean(), 20);
}

}  // namespace test
}  // namespace webrtc
