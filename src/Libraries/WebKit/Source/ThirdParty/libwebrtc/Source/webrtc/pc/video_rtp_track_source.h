/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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
#ifndef PC_VIDEO_RTP_TRACK_SOURCE_H_
#define PC_VIDEO_RTP_TRACK_SOURCE_H_

#include <vector>

#include "api/sequence_checker.h"
#include "api/video/recordable_encoded_frame.h"
#include "api/video/video_frame.h"
#include "api/video/video_sink_interface.h"
#include "api/video/video_source_interface.h"
#include "media/base/video_broadcaster.h"
#include "pc/video_track_source.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/system/no_unique_address.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

// Video track source in use by VideoRtpReceiver
class VideoRtpTrackSource : public VideoTrackSource {
 public:
  class Callback {
   public:
    virtual ~Callback() = default;

    // Called when a keyframe should be generated
    virtual void OnGenerateKeyFrame() = 0;

    // Called when the implementor should eventually start to serve encoded
    // frames using BroadcastEncodedFrameBuffer.
    // The implementor should cause a keyframe to be eventually generated.
    virtual void OnEncodedSinkEnabled(bool enable) = 0;
  };

  explicit VideoRtpTrackSource(Callback* callback);

  VideoRtpTrackSource(const VideoRtpTrackSource&) = delete;
  VideoRtpTrackSource& operator=(const VideoRtpTrackSource&) = delete;

  // Call before the object implementing Callback finishes it's destructor. No
  // more callbacks will be fired after completion. Must be called on the
  // worker thread
  void ClearCallback();

  // Call to broadcast an encoded frame to registered sinks.
  // This method can be called on any thread or queue.
  void BroadcastRecordableEncodedFrame(
      const RecordableEncodedFrame& frame) const;

  // VideoTrackSource
  rtc::VideoSourceInterface<VideoFrame>* source() override;
  rtc::VideoSinkInterface<VideoFrame>* sink();

  // Returns true. This method can be called on any thread.
  bool SupportsEncodedOutput() const override;

  // Generates a key frame. Must be called on the worker thread.
  void GenerateKeyFrame() override;

  // Adds an encoded sink. Must be called on the worker thread.
  void AddEncodedSink(
      rtc::VideoSinkInterface<RecordableEncodedFrame>* sink) override;

  // Removes an encoded sink. Must be called on the worker thread.
  void RemoveEncodedSink(
      rtc::VideoSinkInterface<RecordableEncodedFrame>* sink) override;

 private:
  RTC_NO_UNIQUE_ADDRESS SequenceChecker worker_sequence_checker_{
      SequenceChecker::kDetached};
  // `broadcaster_` is needed since the decoder can only handle one sink.
  // It might be better if the decoder can handle multiple sinks and consider
  // the VideoSinkWants.
  rtc::VideoBroadcaster broadcaster_;
  mutable Mutex mu_;
  std::vector<rtc::VideoSinkInterface<RecordableEncodedFrame>*> encoded_sinks_
      RTC_GUARDED_BY(mu_);
  Callback* callback_ RTC_GUARDED_BY(worker_sequence_checker_);
};

}  // namespace webrtc

#endif  // PC_VIDEO_RTP_TRACK_SOURCE_H_
