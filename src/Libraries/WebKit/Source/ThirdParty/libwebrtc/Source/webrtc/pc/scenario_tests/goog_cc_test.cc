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
#include "api/stats/rtc_stats_collector_callback.h"
#include "api/stats/rtcstats_objects.h"
#include "api/units/data_rate.h"
#include "pc/test/mock_peer_connection_observers.h"
#include "test/field_trial.h"
#include "test/gtest.h"
#include "test/peer_scenario/peer_scenario.h"
#include "test/peer_scenario/peer_scenario_client.h"

namespace webrtc {
namespace test {

// TODO(terelius): Use fake encoder and enable on Android once
// https://bugs.chromium.org/p/webrtc/issues/detail?id=11408 is fixed.
#if defined(WEBRTC_ANDROID)
#define MAYBE_NoBweChangeFromVideoUnmute DISABLED_NoBweChangeFromVideoUnmute
#else
#define MAYBE_NoBweChangeFromVideoUnmute NoBweChangeFromVideoUnmute
#endif
TEST(GoogCcPeerScenarioTest, MAYBE_NoBweChangeFromVideoUnmute) {
  // If transport wide sequence numbers are used for audio, and the call
  // switches from audio only to video only, there will be a sharp change in
  // packets sizes. This will create a change in propagation time which might be
  // detected as an overuse. Using separate overuse detectors for audio and
  // video avoids the issue.
  std::string audio_twcc_trials("WebRTC-Audio-AlrProbing/Disabled/");
  std::string separate_audio_video(
      "WebRTC-Bwe-SeparateAudioPackets/"
      "enabled:true,packet_threshold:15,time_threshold:1000ms/");
  ScopedFieldTrials field_trial(audio_twcc_trials + separate_audio_video);
  PeerScenario s(*test_info_);
  auto* caller = s.CreateClient(PeerScenarioClient::Config());
  auto* callee = s.CreateClient(PeerScenarioClient::Config());

  BuiltInNetworkBehaviorConfig net_conf;
  net_conf.link_capacity = DataRate::KilobitsPerSec(350);
  net_conf.queue_delay_ms = 50;
  auto send_node = s.net()->CreateEmulatedNode(net_conf);
  auto ret_node = s.net()->CreateEmulatedNode(net_conf);

  PeerScenarioClient::VideoSendTrackConfig video_conf;
  video_conf.generator.squares_video->framerate = 15;
  auto video = caller->CreateVideo("VIDEO", video_conf);
  auto audio = caller->CreateAudio("AUDIO", cricket::AudioOptions());

  // Start ICE and exchange SDP.
  s.SimpleConnection(caller, callee, {send_node}, {ret_node});

  // Limit the encoder bitrate to ensure that there are no actual BWE overuses.
  ASSERT_EQ(caller->pc()->GetSenders().size(), 2u);  // 2 senders.
  int num_video_streams = 0;
  for (auto& rtp_sender : caller->pc()->GetSenders()) {
    auto parameters = rtp_sender->GetParameters();
    ASSERT_EQ(parameters.encodings.size(), 1u);  // 1 stream per sender.
    for (auto& encoding_parameters : parameters.encodings) {
      if (encoding_parameters.ssrc == video.sender->ssrc()) {
        num_video_streams++;
        encoding_parameters.max_bitrate_bps = 220000;
        encoding_parameters.max_framerate = 15;
      }
    }
    rtp_sender->SetParameters(parameters);
  }
  ASSERT_EQ(num_video_streams, 1);  // Exactly 1 video stream.

  auto get_bwe = [&] {
    auto callback =
        rtc::make_ref_counted<webrtc::MockRTCStatsCollectorCallback>();
    caller->pc()->GetStats(callback.get());
    s.net()->time_controller()->Wait([&] { return callback->called(); });
    auto stats =
        callback->report()->GetStatsOfType<RTCIceCandidatePairStats>()[0];
    return DataRate::BitsPerSec(*stats->available_outgoing_bitrate);
  };

  s.ProcessMessages(TimeDelta::Seconds(15));
  const DataRate initial_bwe = get_bwe();
  EXPECT_GE(initial_bwe, DataRate::KilobitsPerSec(300));

  // 10 seconds audio only. Bandwidth should not drop.
  video.capturer->Stop();
  s.ProcessMessages(TimeDelta::Seconds(10));
  EXPECT_GE(get_bwe(), initial_bwe);

  // Resume video but stop audio. Bandwidth should not drop.
  video.capturer->Start();
  RTCError status = caller->pc()->RemoveTrackOrError(audio.sender);
  ASSERT_TRUE(status.ok());
  audio.track->set_enabled(false);
  for (int i = 0; i < 10; i++) {
    s.ProcessMessages(TimeDelta::Seconds(1));
    EXPECT_GE(get_bwe(), initial_bwe);
  }

  caller->pc()->Close();
  callee->pc()->Close();
}

}  // namespace test
}  // namespace webrtc
