/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 11, 2025.
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
#include "api/test/create_peer_connection_quality_test_frame_generator.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "api/test/create_frame_generator.h"
#include "api/test/frame_generator_interface.h"
#include "api/test/pclf/media_configuration.h"
#include "api/units/time_delta.h"
#include "rtc_base/checks.h"
#include "system_wrappers/include/clock.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {
namespace webrtc_pc_e2e {

void ValidateScreenShareConfig(const VideoConfig& video_config,
                               const ScreenShareConfig& screen_share_config) {
  if (screen_share_config.slides_yuv_file_names.empty()) {
    if (screen_share_config.scrolling_params) {
      // If we have scrolling params, then its `source_width` and `source_heigh`
      // will be used as width and height of video input, so we have to validate
      // it against width and height of default input.
      RTC_CHECK_EQ(screen_share_config.scrolling_params->source_width,
                   kDefaultSlidesWidth);
      RTC_CHECK_EQ(screen_share_config.scrolling_params->source_height,
                   kDefaultSlidesHeight);
    } else {
      RTC_CHECK_EQ(video_config.width, kDefaultSlidesWidth);
      RTC_CHECK_EQ(video_config.height, kDefaultSlidesHeight);
    }
  }
  if (screen_share_config.scrolling_params) {
    RTC_CHECK_LE(screen_share_config.scrolling_params->duration,
                 screen_share_config.slide_change_interval);
    RTC_CHECK_GE(screen_share_config.scrolling_params->source_width,
                 video_config.width);
    RTC_CHECK_GE(screen_share_config.scrolling_params->source_height,
                 video_config.height);
  }
}

std::unique_ptr<test::FrameGeneratorInterface> CreateSquareFrameGenerator(
    const VideoConfig& video_config,
    std::optional<test::FrameGeneratorInterface::OutputType> type) {
  return test::CreateSquareFrameGenerator(
      video_config.width, video_config.height, std::move(type), std::nullopt);
}

std::unique_ptr<test::FrameGeneratorInterface> CreateFromYuvFileFrameGenerator(
    const VideoConfig& video_config,
    std::string filename) {
  return test::CreateFromYuvFileFrameGenerator(
      {std::move(filename)}, video_config.width, video_config.height,
      /*frame_repeat_count=*/1);
}

std::unique_ptr<test::FrameGeneratorInterface> CreateScreenShareFrameGenerator(
    const VideoConfig& video_config,
    const ScreenShareConfig& screen_share_config) {
  ValidateScreenShareConfig(video_config, screen_share_config);
  if (screen_share_config.generate_slides) {
    return test::CreateSlideFrameGenerator(
        video_config.width, video_config.height,
        screen_share_config.slide_change_interval.seconds() * video_config.fps);
  }
  std::vector<std::string> slides = screen_share_config.slides_yuv_file_names;
  if (slides.empty()) {
    // If slides is empty we need to add default slides as source. In such case
    // video width and height is validated to be equal to kDefaultSlidesWidth
    // and kDefaultSlidesHeight.
    slides.push_back(test::ResourcePath("web_screenshot_1850_1110", "yuv"));
    slides.push_back(test::ResourcePath("presentation_1850_1110", "yuv"));
    slides.push_back(test::ResourcePath("photo_1850_1110", "yuv"));
    slides.push_back(test::ResourcePath("difficult_photo_1850_1110", "yuv"));
  }
  if (!screen_share_config.scrolling_params) {
    // Cycle image every slide_change_interval seconds.
    return test::CreateFromYuvFileFrameGenerator(
        slides, video_config.width, video_config.height,
        screen_share_config.slide_change_interval.seconds() * video_config.fps);
  }

  TimeDelta pause_duration = screen_share_config.slide_change_interval -
                             screen_share_config.scrolling_params->duration;
  RTC_DCHECK(pause_duration >= TimeDelta::Zero());
  return test::CreateScrollingInputFromYuvFilesFrameGenerator(
      Clock::GetRealTimeClock(), slides,
      screen_share_config.scrolling_params->source_width,
      screen_share_config.scrolling_params->source_height, video_config.width,
      video_config.height, screen_share_config.scrolling_params->duration.ms(),
      pause_duration.ms());
}

}  // namespace webrtc_pc_e2e
}  // namespace webrtc
