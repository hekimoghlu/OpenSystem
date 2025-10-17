/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 3, 2023.
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
#include "pc/video_track.h"

#include <memory>

#include "media/base/fake_frame_source.h"
#include "pc/test/fake_video_track_renderer.h"
#include "pc/test/fake_video_track_source.h"
#include "pc/video_track_source.h"
#include "rtc_base/time_utils.h"
#include "test/gtest.h"

using webrtc::FakeVideoTrackRenderer;
using webrtc::FakeVideoTrackSource;
using webrtc::MediaSourceInterface;
using webrtc::MediaStreamTrackInterface;
using webrtc::VideoTrack;
using webrtc::VideoTrackInterface;
using webrtc::VideoTrackSource;

class VideoTrackTest : public ::testing::Test {
 public:
  VideoTrackTest() : frame_source_(640, 480, rtc::kNumMicrosecsPerSec / 30) {
    static const char kVideoTrackId[] = "track_id";
    video_track_source_ = rtc::make_ref_counted<FakeVideoTrackSource>(
        /*is_screencast=*/false);
    video_track_ = VideoTrack::Create(kVideoTrackId, video_track_source_,
                                      rtc::Thread::Current());
  }

 protected:
  rtc::AutoThread main_thread_;
  rtc::scoped_refptr<FakeVideoTrackSource> video_track_source_;
  rtc::scoped_refptr<VideoTrack> video_track_;
  cricket::FakeFrameSource frame_source_;
};

// VideoTrack::Create will create an API proxy around the source object.
// The `GetSource` method provides access to the proxy object intented for API
// use while the GetSourceInternal() provides direct access to the source object
// as provided to the `VideoTrack::Create` factory function.
TEST_F(VideoTrackTest, CheckApiProxyAndInternalSource) {
  EXPECT_NE(video_track_->GetSource(), video_track_source_.get());
  EXPECT_EQ(video_track_->GetSourceInternal(), video_track_source_.get());
}

// Test changing the source state also changes the track state.
TEST_F(VideoTrackTest, SourceStateChangeTrackState) {
  EXPECT_EQ(MediaStreamTrackInterface::kLive, video_track_->state());
  video_track_source_->SetState(MediaSourceInterface::kEnded);
  EXPECT_EQ(MediaStreamTrackInterface::kEnded, video_track_->state());
}

// Test adding renderers to a video track and render to them by providing
// frames to the source.
TEST_F(VideoTrackTest, RenderVideo) {
  // FakeVideoTrackRenderer register itself to `video_track_`
  std::unique_ptr<FakeVideoTrackRenderer> renderer_1(
      new FakeVideoTrackRenderer(video_track_.get()));

  video_track_source_->InjectFrame(frame_source_.GetFrame());
  EXPECT_EQ(1, renderer_1->num_rendered_frames());

  // FakeVideoTrackRenderer register itself to `video_track_`
  std::unique_ptr<FakeVideoTrackRenderer> renderer_2(
      new FakeVideoTrackRenderer(video_track_.get()));
  video_track_source_->InjectFrame(frame_source_.GetFrame());
  EXPECT_EQ(2, renderer_1->num_rendered_frames());
  EXPECT_EQ(1, renderer_2->num_rendered_frames());

  renderer_1.reset(nullptr);
  video_track_source_->InjectFrame(frame_source_.GetFrame());
  EXPECT_EQ(2, renderer_2->num_rendered_frames());
}

// Test that disabling the track results in blacked out frames.
TEST_F(VideoTrackTest, DisableTrackBlackout) {
  std::unique_ptr<FakeVideoTrackRenderer> renderer(
      new FakeVideoTrackRenderer(video_track_.get()));

  video_track_source_->InjectFrame(frame_source_.GetFrame());
  EXPECT_EQ(1, renderer->num_rendered_frames());
  EXPECT_FALSE(renderer->black_frame());

  video_track_->set_enabled(false);
  video_track_source_->InjectFrame(frame_source_.GetFrame());
  EXPECT_EQ(2, renderer->num_rendered_frames());
  EXPECT_TRUE(renderer->black_frame());

  video_track_->set_enabled(true);
  video_track_source_->InjectFrame(frame_source_.GetFrame());
  EXPECT_EQ(3, renderer->num_rendered_frames());
  EXPECT_FALSE(renderer->black_frame());
}
