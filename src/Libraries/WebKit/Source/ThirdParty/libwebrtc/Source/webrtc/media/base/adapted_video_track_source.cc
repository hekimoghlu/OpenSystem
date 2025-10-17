/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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
#include "media/base/adapted_video_track_source.h"

#include "api/scoped_refptr.h"
#include "api/video/i420_buffer.h"
#include "api/video/video_frame_buffer.h"
#include "api/video/video_rotation.h"
#include "rtc_base/checks.h"
#include "rtc_base/time_utils.h"

namespace rtc {

AdaptedVideoTrackSource::AdaptedVideoTrackSource() = default;

AdaptedVideoTrackSource::AdaptedVideoTrackSource(int required_alignment)
    : video_adapter_(required_alignment) {}

AdaptedVideoTrackSource::~AdaptedVideoTrackSource() = default;

bool AdaptedVideoTrackSource::GetStats(Stats* stats) {
  webrtc::MutexLock lock(&stats_mutex_);

  if (!stats_) {
    return false;
  }

  *stats = *stats_;
  return true;
}

void AdaptedVideoTrackSource::OnFrame(const webrtc::VideoFrame& frame) {
  rtc::scoped_refptr<webrtc::VideoFrameBuffer> buffer(
      frame.video_frame_buffer());
  /* Note that this is a "best effort" approach to
     wants.rotation_applied; apply_rotation_ can change from false to
     true between the check of apply_rotation() and the call to
     broadcaster_.OnFrame(), in which case we generate a frame with
     pending rotation despite some sink with wants.rotation_applied ==
     true was just added. The VideoBroadcaster enforces
     synchronization for us in this case, by not passing the frame on
     to sinks which don't want it. */
  if (apply_rotation() && frame.rotation() != webrtc::kVideoRotation_0 &&
      buffer->type() == webrtc::VideoFrameBuffer::Type::kI420) {
    /* Apply pending rotation. */
    webrtc::VideoFrame rotated_frame(frame);
    rotated_frame.set_video_frame_buffer(
        webrtc::I420Buffer::Rotate(*buffer->GetI420(), frame.rotation()));
    rotated_frame.set_rotation(webrtc::kVideoRotation_0);
    broadcaster_.OnFrame(rotated_frame);
  } else {
    broadcaster_.OnFrame(frame);
  }
}

void AdaptedVideoTrackSource::OnFrameDropped() {
  broadcaster_.OnDiscardedFrame();
}

void AdaptedVideoTrackSource::AddOrUpdateSink(
    rtc::VideoSinkInterface<webrtc::VideoFrame>* sink,
    const rtc::VideoSinkWants& wants) {
  broadcaster_.AddOrUpdateSink(sink, wants);
  OnSinkWantsChanged(broadcaster_.wants());
}

void AdaptedVideoTrackSource::RemoveSink(
    rtc::VideoSinkInterface<webrtc::VideoFrame>* sink) {
  broadcaster_.RemoveSink(sink);
  OnSinkWantsChanged(broadcaster_.wants());
}

bool AdaptedVideoTrackSource::apply_rotation() {
  return broadcaster_.wants().rotation_applied;
}

void AdaptedVideoTrackSource::OnSinkWantsChanged(
    const rtc::VideoSinkWants& wants) {
  video_adapter_.OnSinkWants(wants);
}

bool AdaptedVideoTrackSource::AdaptFrame(int width,
                                         int height,
                                         int64_t time_us,
                                         int* out_width,
                                         int* out_height,
                                         int* crop_width,
                                         int* crop_height,
                                         int* crop_x,
                                         int* crop_y) {
  {
    webrtc::MutexLock lock(&stats_mutex_);
    stats_ = Stats{width, height};
  }

  if (!broadcaster_.frame_wanted()) {
    return false;
  }

  if (!video_adapter_.AdaptFrameResolution(
          width, height, time_us * rtc::kNumNanosecsPerMicrosec, crop_width,
          crop_height, out_width, out_height)) {
    broadcaster_.OnDiscardedFrame();
    // VideoAdapter dropped the frame.
    return false;
  }

  *crop_x = (width - *crop_width) / 2;
  *crop_y = (height - *crop_height) / 2;
  return true;
}

void AdaptedVideoTrackSource::ProcessConstraints(
    const webrtc::VideoTrackSourceConstraints& constraints) {
  broadcaster_.ProcessConstraints(constraints);
}

}  // namespace rtc
