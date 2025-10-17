/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 31, 2025.
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
#ifndef TEST_SCENARIO_VIDEO_FRAME_MATCHER_H_
#define TEST_SCENARIO_VIDEO_FRAME_MATCHER_H_

#include <deque>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "api/units/timestamp.h"
#include "api/video/video_frame.h"
#include "api/video/video_sink_interface.h"
#include "api/video/video_source_interface.h"
#include "rtc_base/ref_counted_object.h"
#include "rtc_base/task_queue_for_test.h"
#include "system_wrappers/include/clock.h"
#include "test/scenario/performance_stats.h"

namespace webrtc {
namespace test {

class VideoFrameMatcher {
 public:
  explicit VideoFrameMatcher(
      std::vector<std::function<void(const VideoFramePair&)>>
          frame_pair_handlers);
  ~VideoFrameMatcher();
  void RegisterLayer(int layer_id);
  void OnCapturedFrame(const VideoFrame& frame, Timestamp at_time);
  void OnDecodedFrame(const VideoFrame& frame,
                      int layer_id,
                      Timestamp render_time,
                      Timestamp at_time);
  bool Active() const;

 private:
  struct DecodedFrameBase {
    int id;
    Timestamp decoded_time = Timestamp::PlusInfinity();
    Timestamp render_time = Timestamp::PlusInfinity();
    rtc::scoped_refptr<VideoFrameBuffer> frame;
    rtc::scoped_refptr<VideoFrameBuffer> thumb;
    int repeat_count = 0;
  };
  using DecodedFrame = rtc::FinalRefCountedObject<DecodedFrameBase>;
  struct CapturedFrame {
    int id;
    Timestamp capture_time = Timestamp::PlusInfinity();
    rtc::scoped_refptr<VideoFrameBuffer> frame;
    rtc::scoped_refptr<VideoFrameBuffer> thumb;
    double best_score = INFINITY;
    rtc::scoped_refptr<DecodedFrame> best_decode;
    bool matched = false;
  };
  struct VideoLayer {
    int layer_id;
    std::deque<CapturedFrame> captured_frames;
    rtc::scoped_refptr<DecodedFrame> last_decode;
    int next_decoded_id = 1;
  };
  void HandleMatch(CapturedFrame captured, int layer_id);
  void Finalize();
  int next_capture_id_ = 1;
  std::vector<std::function<void(const VideoFramePair&)>> frame_pair_handlers_;
  std::map<int, VideoLayer> layers_;
  TaskQueueForTest task_queue_;
};

class CapturedFrameTap : public rtc::VideoSinkInterface<VideoFrame> {
 public:
  CapturedFrameTap(Clock* clock, VideoFrameMatcher* matcher);
  CapturedFrameTap(CapturedFrameTap&) = delete;
  CapturedFrameTap& operator=(CapturedFrameTap&) = delete;

  void OnFrame(const VideoFrame& frame) override;
  void OnDiscardedFrame() override;

 private:
  Clock* const clock_;
  VideoFrameMatcher* const matcher_;
  int discarded_count_ = 0;
};

class ForwardingCapturedFrameTap
    : public rtc::VideoSinkInterface<VideoFrame>,
      public rtc::VideoSourceInterface<VideoFrame> {
 public:
  ForwardingCapturedFrameTap(Clock* clock,
                             VideoFrameMatcher* matcher,
                             rtc::VideoSourceInterface<VideoFrame>* source);
  ForwardingCapturedFrameTap(ForwardingCapturedFrameTap&) = delete;
  ForwardingCapturedFrameTap& operator=(ForwardingCapturedFrameTap&) = delete;

  // VideoSinkInterface interface
  void OnFrame(const VideoFrame& frame) override;
  void OnDiscardedFrame() override;

  // VideoSourceInterface interface
  void AddOrUpdateSink(VideoSinkInterface<VideoFrame>* sink,
                       const rtc::VideoSinkWants& wants) override;
  void RemoveSink(VideoSinkInterface<VideoFrame>* sink) override;

 private:
  Clock* const clock_;
  VideoFrameMatcher* const matcher_;
  rtc::VideoSourceInterface<VideoFrame>* const source_;
  VideoSinkInterface<VideoFrame>* sink_ = nullptr;
  int discarded_count_ = 0;
};

class DecodedFrameTap : public rtc::VideoSinkInterface<VideoFrame> {
 public:
  DecodedFrameTap(Clock* clock, VideoFrameMatcher* matcher, int layer_id);
  // VideoSinkInterface interface
  void OnFrame(const VideoFrame& frame) override;

 private:
  Clock* const clock_;
  VideoFrameMatcher* const matcher_;
  int layer_id_;
};
}  // namespace test
}  // namespace webrtc
#endif  // TEST_SCENARIO_VIDEO_FRAME_MATCHER_H_
