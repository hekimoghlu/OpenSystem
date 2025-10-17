/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 16, 2021.
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
#include "test/pc/e2e/media/media_helper.h"

#include <string>
#include <utility>

#include "absl/types/variant.h"
#include "api/media_stream_interface.h"
#include "api/test/create_frame_generator.h"
#include "api/test/pclf/media_configuration.h"
#include "api/test/pclf/peer_configurer.h"
#include "test/frame_generator_capturer.h"
#include "test/platform_video_capturer.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {
namespace webrtc_pc_e2e {

void MediaHelper::MaybeAddAudio(TestPeer* peer) {
  if (!peer->params().audio_config) {
    return;
  }
  const AudioConfig& audio_config = peer->params().audio_config.value();
  rtc::scoped_refptr<webrtc::AudioSourceInterface> source =
      peer->pc_factory()->CreateAudioSource(audio_config.audio_options);
  rtc::scoped_refptr<AudioTrackInterface> track =
      peer->pc_factory()->CreateAudioTrack(*audio_config.stream_label,
                                           source.get());
  std::string sync_group = audio_config.sync_group
                               ? audio_config.sync_group.value()
                               : audio_config.stream_label.value() + "-sync";
  peer->AddTrack(track, {sync_group, *audio_config.stream_label});
}

std::vector<rtc::scoped_refptr<TestVideoCapturerVideoTrackSource>>
MediaHelper::MaybeAddVideo(TestPeer* peer) {
  // Params here valid because of pre-run validation.
  const Params& params = peer->params();
  const ConfigurableParams& configurable_params = peer->configurable_params();
  std::vector<rtc::scoped_refptr<TestVideoCapturerVideoTrackSource>> out;
  for (size_t i = 0; i < configurable_params.video_configs.size(); ++i) {
    const VideoConfig& video_config = configurable_params.video_configs[i];
    // Setup input video source into peer connection.
    std::unique_ptr<test::TestVideoCapturer> capturer = CreateVideoCapturer(
        video_config, peer->ReleaseVideoSource(i),
        video_quality_analyzer_injection_helper_->CreateFramePreprocessor(
            params.name.value(), video_config));
    bool is_screencast =
        video_config.content_hint == VideoTrackInterface::ContentHint::kText ||
        video_config.content_hint ==
            VideoTrackInterface::ContentHint::kDetailed;
    rtc::scoped_refptr<TestVideoCapturerVideoTrackSource> source =
        rtc::make_ref_counted<TestVideoCapturerVideoTrackSource>(
            std::move(capturer), is_screencast, video_config.stream_label);
    out.push_back(source);
    RTC_LOG(LS_INFO) << "Adding video with video_config.stream_label="
                     << video_config.stream_label.value();
    rtc::scoped_refptr<VideoTrackInterface> track =
        peer->pc_factory()->CreateVideoTrack(source,
                                             video_config.stream_label.value());
    if (video_config.content_hint.has_value()) {
      track->set_content_hint(video_config.content_hint.value());
    }
    std::string sync_group = video_config.sync_group
                                 ? video_config.sync_group.value()
                                 : video_config.stream_label.value() + "-sync";
    RTCErrorOr<rtc::scoped_refptr<RtpSenderInterface>> sender =
        peer->AddTrack(track, {sync_group, *video_config.stream_label});
    RTC_CHECK(sender.ok());
    if (video_config.temporal_layers_count ||
        video_config.degradation_preference) {
      RtpParameters rtp_parameters = sender.value()->GetParameters();
      if (video_config.temporal_layers_count) {
        for (auto& encoding_parameters : rtp_parameters.encodings) {
          encoding_parameters.num_temporal_layers =
              video_config.temporal_layers_count;
        }
      }
      if (video_config.degradation_preference) {
        rtp_parameters.degradation_preference =
            video_config.degradation_preference;
      }
      RTCError res = sender.value()->SetParameters(rtp_parameters);
      RTC_CHECK(res.ok()) << "Failed to set RTP parameters";
    }
  }
  return out;
}

std::unique_ptr<test::TestVideoCapturer> MediaHelper::CreateVideoCapturer(
    const VideoConfig& video_config,
    PeerConfigurer::VideoSource source,
    std::unique_ptr<test::TestVideoCapturer::FramePreprocessor>
        frame_preprocessor) {
  CapturingDeviceIndex* capturing_device_index =
      absl::get_if<CapturingDeviceIndex>(&source);
  if (capturing_device_index != nullptr) {
    std::unique_ptr<test::TestVideoCapturer> capturer =
        test::CreateVideoCapturer(video_config.width, video_config.height,
                                  video_config.fps,
                                  static_cast<size_t>(*capturing_device_index));
    RTC_CHECK(capturer)
        << "Failed to obtain input stream from capturing device #"
        << *capturing_device_index;
    capturer->SetFramePreprocessor(std::move(frame_preprocessor));
    return capturer;
  }

  auto capturer = std::make_unique<test::FrameGeneratorCapturer>(
      clock_,
      absl::get<std::unique_ptr<test::FrameGeneratorInterface>>(
          std::move(source)),
      video_config.fps, *task_queue_factory_);
  capturer->SetFramePreprocessor(std::move(frame_preprocessor));
  capturer->Init();
  return capturer;
}

}  // namespace webrtc_pc_e2e
}  // namespace webrtc
