/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 7, 2022.
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
#include "video/render/incoming_video_stream.h"

#include <memory>
#include <optional>
#include <utility>

#include "api/units/time_delta.h"
#include "rtc_base/checks.h"
#include "rtc_base/trace_event.h"
#include "video/render/video_render_frames.h"

namespace webrtc {

IncomingVideoStream::IncomingVideoStream(
    TaskQueueFactory* task_queue_factory,
    int32_t delay_ms,
    rtc::VideoSinkInterface<VideoFrame>* callback)
    : render_buffers_(delay_ms),
      callback_(callback),
      incoming_render_queue_(task_queue_factory->CreateTaskQueue(
          "IncomingVideoStream",
          TaskQueueFactory::Priority::HIGH)) {}

IncomingVideoStream::~IncomingVideoStream() {
  RTC_DCHECK(main_thread_checker_.IsCurrent());
  // The queue must be destroyed before its pointer is invalidated to avoid race
  // between destructor and posting task to the task queue from itself.
  // std::unique_ptr destructor does the same two operations in reverse order as
  // it doesn't expect member would be used after its destruction has started.
  incoming_render_queue_.get_deleter()(incoming_render_queue_.get());
  incoming_render_queue_.release();
}

void IncomingVideoStream::OnFrame(const VideoFrame& video_frame) {
  TRACE_EVENT0("webrtc", "IncomingVideoStream::OnFrame");
  RTC_CHECK_RUNS_SERIALIZED(&decoder_race_checker_);
  RTC_DCHECK(!incoming_render_queue_->IsCurrent());
  // TODO(srte): Using video_frame = std::move(video_frame) would move the frame
  // into the lambda instead of copying it, but it doesn't work unless we change
  // OnFrame to take its frame argument by value instead of const reference.
  incoming_render_queue_->PostTask([this, video_frame = video_frame]() mutable {
    RTC_DCHECK_RUN_ON(incoming_render_queue_.get());
    if (render_buffers_.AddFrame(std::move(video_frame)) == 1)
      Dequeue();
  });
}

void IncomingVideoStream::Dequeue() {
  TRACE_EVENT0("webrtc", "IncomingVideoStream::Dequeue");
  RTC_DCHECK_RUN_ON(incoming_render_queue_.get());
  std::optional<VideoFrame> frame_to_render = render_buffers_.FrameToRender();
  if (frame_to_render)
    callback_->OnFrame(*frame_to_render);

  if (render_buffers_.HasPendingFrames()) {
    uint32_t wait_time = render_buffers_.TimeToNextFrameRelease();
    incoming_render_queue_->PostDelayedHighPrecisionTask(
        [this]() { Dequeue(); }, TimeDelta::Millis(wait_time));
  }
}

}  // namespace webrtc
