/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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
#ifndef VIDEO_RENDER_INCOMING_VIDEO_STREAM_H_
#define VIDEO_RENDER_INCOMING_VIDEO_STREAM_H_

#include <stdint.h>

#include <memory>

#include "api/sequence_checker.h"
#include "api/task_queue/task_queue_base.h"
#include "api/task_queue/task_queue_factory.h"
#include "api/video/video_frame.h"
#include "api/video/video_sink_interface.h"
#include "rtc_base/race_checker.h"
#include "rtc_base/thread_annotations.h"
#include "video/render/video_render_frames.h"

namespace webrtc {

class IncomingVideoStream : public rtc::VideoSinkInterface<VideoFrame> {
 public:
  IncomingVideoStream(TaskQueueFactory* task_queue_factory,
                      int32_t delay_ms,
                      rtc::VideoSinkInterface<VideoFrame>* callback);
  ~IncomingVideoStream() override;

 private:
  void OnFrame(const VideoFrame& video_frame) override;
  void Dequeue();

  SequenceChecker main_thread_checker_;
  rtc::RaceChecker decoder_race_checker_;

  VideoRenderFrames render_buffers_ RTC_GUARDED_BY(incoming_render_queue_);
  rtc::VideoSinkInterface<VideoFrame>* const callback_;
  std::unique_ptr<TaskQueueBase, TaskQueueDeleter> incoming_render_queue_;
};

}  // namespace webrtc

#endif  // VIDEO_RENDER_INCOMING_VIDEO_STREAM_H_
