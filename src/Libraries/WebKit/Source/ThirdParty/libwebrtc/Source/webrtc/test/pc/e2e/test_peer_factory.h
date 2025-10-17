/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 8, 2024.
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
#ifndef TEST_PC_E2E_TEST_PEER_FACTORY_H_
#define TEST_PC_E2E_TEST_PEER_FACTORY_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/base/macros.h"
#include "api/task_queue/task_queue_base.h"
#include "api/test/pclf/media_configuration.h"
#include "api/test/pclf/peer_configurer.h"
#include "api/test/time_controller.h"
#include "pc/test/mock_peer_connection_observers.h"
#include "rtc_base/thread.h"
#include "test/pc/e2e/analyzer/video/video_quality_analyzer_injection_helper.h"
#include "test/pc/e2e/test_peer.h"

namespace webrtc {
namespace webrtc_pc_e2e {

struct RemotePeerAudioConfig {
  explicit RemotePeerAudioConfig(AudioConfig config)
      : sampling_frequency_in_hz(config.sampling_frequency_in_hz),
        output_file_name(config.output_dump_file_name) {}

  static std::optional<RemotePeerAudioConfig> Create(
      std::optional<AudioConfig> config);

  int sampling_frequency_in_hz;
  std::optional<std::string> output_file_name;
};

class TestPeerFactory {
 public:
  // Creates a test peer factory.
  // `signaling_thread` will be used as a signaling thread for all peers created
  // by this factory.
  // `time_controller` will be used to create required threads, task queue
  // factories and call factory.
  // `video_analyzer_helper` will be used to setup video quality analysis for
  // created peers.
  TestPeerFactory(rtc::Thread* signaling_thread,
                  TimeController& time_controller,
                  VideoQualityAnalyzerInjectionHelper* video_analyzer_helper)
      : signaling_thread_(signaling_thread),
        time_controller_(time_controller),
        video_analyzer_helper_(video_analyzer_helper) {}

  ABSL_DEPRECATE_AND_INLINE()
  TestPeerFactory(rtc::Thread* signaling_thread,
                  TimeController& time_controller,
                  VideoQualityAnalyzerInjectionHelper* video_analyzer_helper,
                  TaskQueueBase* /*task_queue*/)
      : TestPeerFactory(signaling_thread,
                        time_controller,
                        video_analyzer_helper) {}

  // Setups all components, that should be provided to WebRTC
  // PeerConnectionFactory and PeerConnection creation methods,
  // also will setup dependencies, that are required for media analyzers
  // injection.
  std::unique_ptr<TestPeer> CreateTestPeer(
      std::unique_ptr<PeerConfigurer> configurer,
      std::unique_ptr<MockPeerConnectionObserver> observer,
      std::optional<RemotePeerAudioConfig> remote_audio_config,
      std::optional<EchoEmulationConfig> echo_emulation_config);

 private:
  rtc::Thread* signaling_thread_;
  TimeController& time_controller_;
  VideoQualityAnalyzerInjectionHelper* video_analyzer_helper_;
};

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // TEST_PC_E2E_TEST_PEER_FACTORY_H_
