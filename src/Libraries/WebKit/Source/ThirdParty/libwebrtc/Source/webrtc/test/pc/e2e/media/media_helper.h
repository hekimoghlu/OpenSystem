/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 4, 2023.
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
#ifndef TEST_PC_E2E_MEDIA_MEDIA_HELPER_H_
#define TEST_PC_E2E_MEDIA_MEDIA_HELPER_H_

#include <memory>
#include <vector>

#include "api/test/frame_generator_interface.h"
#include "api/test/pclf/media_configuration.h"
#include "api/test/pclf/peer_configurer.h"
#include "test/pc/e2e/analyzer/video/video_quality_analyzer_injection_helper.h"
#include "test/pc/e2e/media/test_video_capturer_video_track_source.h"
#include "test/pc/e2e/test_peer.h"

namespace webrtc {
namespace webrtc_pc_e2e {

class MediaHelper {
 public:
  MediaHelper(VideoQualityAnalyzerInjectionHelper*
                  video_quality_analyzer_injection_helper,
              TaskQueueFactory* task_queue_factory,
              Clock* clock)
      : clock_(clock),
        task_queue_factory_(task_queue_factory),
        video_quality_analyzer_injection_helper_(
            video_quality_analyzer_injection_helper) {}

  void MaybeAddAudio(TestPeer* peer);

  std::vector<rtc::scoped_refptr<TestVideoCapturerVideoTrackSource>>
  MaybeAddVideo(TestPeer* peer);

 private:
  std::unique_ptr<test::TestVideoCapturer> CreateVideoCapturer(
      const VideoConfig& video_config,
      PeerConfigurer::VideoSource source,
      std::unique_ptr<test::TestVideoCapturer::FramePreprocessor>
          frame_preprocessor);

  Clock* const clock_;
  TaskQueueFactory* const task_queue_factory_;
  VideoQualityAnalyzerInjectionHelper* video_quality_analyzer_injection_helper_;
};

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // TEST_PC_E2E_MEDIA_MEDIA_HELPER_H_
