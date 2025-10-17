/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 6, 2025.
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
#include "test/create_frame_generator_capturer.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "api/task_queue/task_queue_factory.h"
#include "api/test/create_frame_generator.h"
#include "api/test/frame_generator_interface.h"
#include "api/units/time_delta.h"
#include "rtc_base/checks.h"
#include "system_wrappers/include/clock.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {
namespace test {
namespace {

std::string TransformFilePath(std::string path) {
  static const std::string resource_prefix = "res://";
  int ext_pos = path.rfind('.');
  if (ext_pos < 0) {
    return test::ResourcePath(path, "yuv");
  } else if (absl::StartsWith(path, resource_prefix)) {
    std::string name = path.substr(resource_prefix.length(), ext_pos);
    std::string ext = path.substr(ext_pos, path.size());
    return test::ResourcePath(name, ext);
  }
  return path;
}

}  // namespace

std::unique_ptr<FrameGeneratorCapturer> CreateFrameGeneratorCapturer(
    Clock* clock,
    TaskQueueFactory& task_queue_factory,
    FrameGeneratorCapturerConfig::SquaresVideo config) {
  return std::make_unique<FrameGeneratorCapturer>(
      clock,
      CreateSquareFrameGenerator(config.width, config.height,
                                 config.pixel_format, config.num_squares),
      config.framerate, task_queue_factory);
}
std::unique_ptr<FrameGeneratorCapturer> CreateFrameGeneratorCapturer(
    Clock* clock,
    TaskQueueFactory& task_queue_factory,
    FrameGeneratorCapturerConfig::SquareSlides config) {
  return std::make_unique<FrameGeneratorCapturer>(
      clock,
      CreateSlideFrameGenerator(
          config.width, config.height,
          /*frame_repeat_count*/ config.change_interval.seconds<double>() *
              config.framerate),
      config.framerate, task_queue_factory);
}
std::unique_ptr<FrameGeneratorCapturer> CreateFrameGeneratorCapturer(
    Clock* clock,
    TaskQueueFactory& task_queue_factory,
    FrameGeneratorCapturerConfig::VideoFile config) {
  RTC_CHECK(config.width && config.height);
  return std::make_unique<FrameGeneratorCapturer>(
      clock,
      CreateFromYuvFileFrameGenerator({TransformFilePath(config.name)},
                                      config.width, config.height,
                                      /*frame_repeat_count*/ 1),
      config.framerate, task_queue_factory);
}

std::unique_ptr<FrameGeneratorCapturer> CreateFrameGeneratorCapturer(
    Clock* clock,
    TaskQueueFactory& task_queue_factory,
    FrameGeneratorCapturerConfig::ImageSlides config) {
  std::unique_ptr<FrameGeneratorInterface> slides_generator;
  std::vector<std::string> paths = config.paths;
  for (std::string& path : paths)
    path = TransformFilePath(path);

  if (config.crop.width || config.crop.height) {
    TimeDelta pause_duration =
        config.change_interval - config.crop.scroll_duration;
    RTC_CHECK_GE(pause_duration, TimeDelta::Zero());
    int crop_width = config.crop.width.value_or(config.width);
    int crop_height = config.crop.height.value_or(config.height);
    RTC_CHECK_LE(crop_width, config.width);
    RTC_CHECK_LE(crop_height, config.height);
    slides_generator = CreateScrollingInputFromYuvFilesFrameGenerator(
        clock, paths, config.width, config.height, crop_width, crop_height,
        config.crop.scroll_duration.ms(), pause_duration.ms());
  } else {
    slides_generator = CreateFromYuvFileFrameGenerator(
        paths, config.width, config.height,
        /*frame_repeat_count*/ config.change_interval.seconds<double>() *
            config.framerate);
  }
  return std::make_unique<FrameGeneratorCapturer>(
      clock, std::move(slides_generator), config.framerate, task_queue_factory);
}

std::unique_ptr<FrameGeneratorCapturer> CreateFrameGeneratorCapturer(
    Clock* clock,
    TaskQueueFactory& task_queue_factory,
    const FrameGeneratorCapturerConfig& config) {
  if (config.video_file) {
    return CreateFrameGeneratorCapturer(clock, task_queue_factory,
                                        *config.video_file);
  } else if (config.image_slides) {
    return CreateFrameGeneratorCapturer(clock, task_queue_factory,
                                        *config.image_slides);
  } else if (config.squares_slides) {
    return CreateFrameGeneratorCapturer(clock, task_queue_factory,
                                        *config.squares_slides);
  } else {
    return CreateFrameGeneratorCapturer(
        clock, task_queue_factory,
        config.squares_video.value_or(
            FrameGeneratorCapturerConfig::SquaresVideo()));
  }
}

}  // namespace test
}  // namespace webrtc
