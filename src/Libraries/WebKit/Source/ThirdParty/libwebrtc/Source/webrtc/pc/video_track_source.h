/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 12, 2025.
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
#ifndef PC_VIDEO_TRACK_SOURCE_H_
#define PC_VIDEO_TRACK_SOURCE_H_

#include <optional>

#include "api/media_stream_interface.h"
#include "api/notifier.h"
#include "api/sequence_checker.h"
#include "api/video/recordable_encoded_frame.h"
#include "api/video/video_frame.h"
#include "api/video/video_sink_interface.h"
#include "api/video/video_source_interface.h"
#include "media/base/media_channel.h"
#include "rtc_base/checks.h"
#include "rtc_base/system/no_unique_address.h"
#include "rtc_base/system/rtc_export.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

// VideoTrackSource is a convenience base class for implementations of
// VideoTrackSourceInterface.
class RTC_EXPORT VideoTrackSource : public Notifier<VideoTrackSourceInterface> {
 public:
  explicit VideoTrackSource(bool remote);
  void SetState(SourceState new_state);

  SourceState state() const override {
    RTC_DCHECK_RUN_ON(&signaling_thread_checker_);
    return state_;
  }
  bool remote() const override { return remote_; }

  bool is_screencast() const override { return false; }
  std::optional<bool> needs_denoising() const override { return std::nullopt; }

  bool GetStats(Stats* stats) override { return false; }

  void AddOrUpdateSink(rtc::VideoSinkInterface<VideoFrame>* sink,
                       const rtc::VideoSinkWants& wants) override;
  void RemoveSink(rtc::VideoSinkInterface<VideoFrame>* sink) override;

  bool SupportsEncodedOutput() const override { return false; }
  void GenerateKeyFrame() override {}
  void AddEncodedSink(
      rtc::VideoSinkInterface<RecordableEncodedFrame>* sink) override {}
  void RemoveEncodedSink(
      rtc::VideoSinkInterface<RecordableEncodedFrame>* sink) override {}

 protected:
  virtual rtc::VideoSourceInterface<VideoFrame>* source() = 0;

 private:
  RTC_NO_UNIQUE_ADDRESS SequenceChecker worker_thread_checker_{
      SequenceChecker::kDetached};
  RTC_NO_UNIQUE_ADDRESS SequenceChecker signaling_thread_checker_;
  SourceState state_ RTC_GUARDED_BY(&signaling_thread_checker_);
  const bool remote_;
};

}  // namespace webrtc

#endif  //  PC_VIDEO_TRACK_SOURCE_H_
