/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 17, 2023.
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
#ifndef VIDEO_VIDEO_SOURCE_SINK_CONTROLLER_H_
#define VIDEO_VIDEO_SOURCE_SINK_CONTROLLER_H_

#include <optional>
#include <string>
#include <vector>

#include "api/sequence_checker.h"
#include "api/video/video_frame.h"
#include "api/video/video_sink_interface.h"
#include "api/video/video_source_interface.h"
#include "call/adaptation/video_source_restrictions.h"
#include "rtc_base/system/no_unique_address.h"

namespace webrtc {

// Responsible for configuring source/sink settings, i.e. performing
// rtc::VideoSourceInterface<VideoFrame>::AddOrUpdateSink(). It does this by
// storing settings internally which are converted to rtc::VideoSinkWants when
// PushSourceSinkSettings() is performed.
class VideoSourceSinkController {
 public:
  VideoSourceSinkController(rtc::VideoSinkInterface<VideoFrame>* sink,
                            rtc::VideoSourceInterface<VideoFrame>* source);

  ~VideoSourceSinkController();

  void SetSource(rtc::VideoSourceInterface<VideoFrame>* source);
  bool HasSource() const;

  // Requests a refresh frame from the current source, if set.
  void RequestRefreshFrame();

  // Must be called in order for changes to settings to have an effect. This
  // allows you to modify multiple properties in a single push to the sink.
  void PushSourceSinkSettings();

  VideoSourceRestrictions restrictions() const;
  std::optional<size_t> pixels_per_frame_upper_limit() const;
  std::optional<double> frame_rate_upper_limit() const;
  bool rotation_applied() const;
  int resolution_alignment() const;
  const std::vector<rtc::VideoSinkWants::FrameSize>& resolutions() const;
  bool active() const;
  std::optional<rtc::VideoSinkWants::FrameSize> scale_resolution_down_to()
      const;

  // Updates the settings stored internally. In order for these settings to be
  // applied to the sink, PushSourceSinkSettings() must subsequently be called.
  void SetRestrictions(VideoSourceRestrictions restrictions);
  void SetPixelsPerFrameUpperLimit(
      std::optional<size_t> pixels_per_frame_upper_limit);
  void SetFrameRateUpperLimit(std::optional<double> frame_rate_upper_limit);
  void SetRotationApplied(bool rotation_applied);
  void SetResolutionAlignment(int resolution_alignment);
  void SetResolutions(std::vector<rtc::VideoSinkWants::FrameSize> resolutions);
  void SetActive(bool active);
  void SetScaleResolutionDownTo(
      std::optional<rtc::VideoSinkWants::FrameSize> scale_resolution_down_to);

 private:
  rtc::VideoSinkWants CurrentSettingsToSinkWants() const
      RTC_EXCLUSIVE_LOCKS_REQUIRED(sequence_checker_);

  // Used to ensure that this class is called on threads/sequences that it and
  // downstream implementations were designed for.
  // In practice, this represent's libjingle's worker thread.
  RTC_NO_UNIQUE_ADDRESS SequenceChecker sequence_checker_;

  rtc::VideoSinkInterface<VideoFrame>* const sink_;
  rtc::VideoSourceInterface<VideoFrame>* source_
      RTC_GUARDED_BY(&sequence_checker_);
  // Pixel and frame rate restrictions.
  VideoSourceRestrictions restrictions_ RTC_GUARDED_BY(&sequence_checker_);
  // Ensures that even if we are not restricted, the sink is never configured
  // above this limit. Example: We are not CPU limited (no `restrictions_`) but
  // our encoder is capped at 30 fps (= `frame_rate_upper_limit_`).
  std::optional<size_t> pixels_per_frame_upper_limit_
      RTC_GUARDED_BY(&sequence_checker_);
  std::optional<double> frame_rate_upper_limit_
      RTC_GUARDED_BY(&sequence_checker_);
  bool rotation_applied_ RTC_GUARDED_BY(&sequence_checker_) = false;
  int resolution_alignment_ RTC_GUARDED_BY(&sequence_checker_) = 1;
  std::vector<rtc::VideoSinkWants::FrameSize> resolutions_
      RTC_GUARDED_BY(&sequence_checker_);
  bool active_ RTC_GUARDED_BY(&sequence_checker_) = true;
  std::optional<rtc::VideoSinkWants::FrameSize> scale_resolution_down_to_
      RTC_GUARDED_BY(&sequence_checker_);
};

}  // namespace webrtc

#endif  // VIDEO_VIDEO_SOURCE_SINK_CONTROLLER_H_
