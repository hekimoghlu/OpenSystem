/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 2, 2025.
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
#ifndef MEDIA_BASE_VIDEO_SOURCE_BASE_H_
#define MEDIA_BASE_VIDEO_SOURCE_BASE_H_

#include <vector>

#include "api/sequence_checker.h"
#include "api/video/video_frame.h"
#include "api/video/video_sink_interface.h"
#include "api/video/video_source_interface.h"
#include "rtc_base/system/no_unique_address.h"

namespace rtc {

// VideoSourceBase is not thread safe. Before using this class, consider using
// VideoSourceBaseGuarded below instead, which is an identical implementation
// but applies a sequence checker to help protect internal state.
// TODO(bugs.webrtc.org/12780): Delete this class.
class VideoSourceBase : public VideoSourceInterface<webrtc::VideoFrame> {
 public:
  VideoSourceBase();
  ~VideoSourceBase() override;
  void AddOrUpdateSink(VideoSinkInterface<webrtc::VideoFrame>* sink,
                       const VideoSinkWants& wants) override;
  void RemoveSink(VideoSinkInterface<webrtc::VideoFrame>* sink) override;

 protected:
  struct SinkPair {
    SinkPair(VideoSinkInterface<webrtc::VideoFrame>* sink, VideoSinkWants wants)
        : sink(sink), wants(wants) {}
    VideoSinkInterface<webrtc::VideoFrame>* sink;
    VideoSinkWants wants;
  };
  SinkPair* FindSinkPair(const VideoSinkInterface<webrtc::VideoFrame>* sink);

  const std::vector<SinkPair>& sink_pairs() const { return sinks_; }

 private:
  std::vector<SinkPair> sinks_;
};

// VideoSourceBaseGuarded assumes that operations related to sinks, occur on the
// same TQ/thread that the object was constructed on.
class VideoSourceBaseGuarded : public VideoSourceInterface<webrtc::VideoFrame> {
 public:
  VideoSourceBaseGuarded();
  ~VideoSourceBaseGuarded() override;

  void AddOrUpdateSink(VideoSinkInterface<webrtc::VideoFrame>* sink,
                       const VideoSinkWants& wants) override;
  void RemoveSink(VideoSinkInterface<webrtc::VideoFrame>* sink) override;

 protected:
  struct SinkPair {
    SinkPair(VideoSinkInterface<webrtc::VideoFrame>* sink, VideoSinkWants wants)
        : sink(sink), wants(wants) {}
    VideoSinkInterface<webrtc::VideoFrame>* sink;
    VideoSinkWants wants;
  };

  SinkPair* FindSinkPair(const VideoSinkInterface<webrtc::VideoFrame>* sink);
  const std::vector<SinkPair>& sink_pairs() const;

  // Keep the `source_sequence_` checker protected to allow sub classes the
  // ability to call Detach() if/when appropriate.
  RTC_NO_UNIQUE_ADDRESS webrtc::SequenceChecker source_sequence_;

 private:
  std::vector<SinkPair> sinks_ RTC_GUARDED_BY(&source_sequence_);
};

}  // namespace rtc

#endif  // MEDIA_BASE_VIDEO_SOURCE_BASE_H_
