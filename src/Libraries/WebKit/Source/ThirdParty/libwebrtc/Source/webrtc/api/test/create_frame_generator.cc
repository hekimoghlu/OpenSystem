/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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
#include "api/test/create_frame_generator.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/strings/string_view.h"
#include "api/environment/environment.h"
#include "api/environment/environment_factory.h"
#include "api/test/frame_generator_interface.h"
#include "rtc_base/checks.h"
#include "test/frame_generator.h"
#include "test/testsupport/ivf_video_frame_generator.h"

namespace webrtc {
namespace test {

std::unique_ptr<FrameGeneratorInterface> CreateSquareFrameGenerator(
    int width,
    int height,
    std::optional<FrameGeneratorInterface::OutputType> type,
    std::optional<int> num_squares) {
  return std::make_unique<SquareGenerator>(
      width, height, type.value_or(FrameGeneratorInterface::OutputType::kI420),
      num_squares.value_or(10));
}

std::unique_ptr<FrameGeneratorInterface> CreateFromYuvFileFrameGenerator(
    std::vector<std::string> filenames,
    size_t width,
    size_t height,
    int frame_repeat_count) {
  RTC_DCHECK(!filenames.empty());
  std::vector<FILE*> files;
  for (const std::string& filename : filenames) {
    FILE* file = fopen(filename.c_str(), "rb");
    RTC_DCHECK(file != nullptr) << "Failed to open: '" << filename << "'\n";
    files.push_back(file);
  }

  return std::make_unique<YuvFileGenerator>(files, width, height,
                                            frame_repeat_count);
}

std::unique_ptr<FrameGeneratorInterface> CreateFromNV12FileFrameGenerator(
    std::vector<std::string> filenames,
    size_t width,
    size_t height,
    int frame_repeat_count) {
  RTC_DCHECK(!filenames.empty());
  std::vector<FILE*> files;
  for (const std::string& filename : filenames) {
    FILE* file = fopen(filename.c_str(), "rb");
    RTC_DCHECK(file != nullptr) << "Failed to open: '" << filename << "'\n";
    files.push_back(file);
  }

  return std::make_unique<NV12FileGenerator>(files, width, height,
                                             frame_repeat_count);
}

std::unique_ptr<FrameGeneratorInterface> CreateFromIvfFileFrameGenerator(
    std::string filename) {
  return CreateFromIvfFileFrameGenerator(CreateEnvironment(), filename);
}

absl::Nonnull<std::unique_ptr<FrameGeneratorInterface>>
CreateFromIvfFileFrameGenerator(const Environment& env,
                                absl::string_view filename) {
  return std::make_unique<IvfVideoFrameGenerator>(env, filename);
}

std::unique_ptr<FrameGeneratorInterface>
CreateScrollingInputFromYuvFilesFrameGenerator(
    Clock* clock,
    std::vector<std::string> filenames,
    size_t source_width,
    size_t source_height,
    size_t target_width,
    size_t target_height,
    int64_t scroll_time_ms,
    int64_t pause_time_ms) {
  RTC_DCHECK(!filenames.empty());
  std::vector<FILE*> files;
  for (const std::string& filename : filenames) {
    FILE* file = fopen(filename.c_str(), "rb");
    RTC_DCHECK(file != nullptr);
    files.push_back(file);
  }

  return std::make_unique<ScrollingImageFrameGenerator>(
      clock, files, source_width, source_height, target_width, target_height,
      scroll_time_ms, pause_time_ms);
}

std::unique_ptr<FrameGeneratorInterface>
CreateSlideFrameGenerator(int width, int height, int frame_repeat_count) {
  return std::make_unique<SlideGenerator>(width, height, frame_repeat_count);
}

}  // namespace test
}  // namespace webrtc
