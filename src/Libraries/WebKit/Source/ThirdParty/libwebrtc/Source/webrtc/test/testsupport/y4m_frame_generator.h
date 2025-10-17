/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 27, 2022.
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
#ifndef TEST_TESTSUPPORT_Y4M_FRAME_GENERATOR_H_
#define TEST_TESTSUPPORT_Y4M_FRAME_GENERATOR_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "api/test/frame_generator_interface.h"
#include "rtc_base/checks.h"
#include "test/testsupport/frame_reader.h"

namespace webrtc {
namespace test {

// Generates frames from a Y4M file. The behaviour when reaching EOF is
// configurable via RepeatMode.
class Y4mFrameGenerator : public FrameGeneratorInterface {
 public:
  enum class RepeatMode {
    // Generate frames from the input file, but it stops generating new frames
    // once EOF is reached.
    kSingle,
    // Generate frames from the input file, when EOF is reached it starts from
    // the beginning.
    kLoop,
    // Generate frames from the input file, when EOF is reached it plays frames
    // backwards from the end to the beginning of the file (and vice versa,
    // literally doing Ping/Pong between the beginning and the end of the file).
    kPingPong,
  };
  Y4mFrameGenerator(absl::string_view filename, RepeatMode repeat_mode);
  ~Y4mFrameGenerator() override = default;

  VideoFrameData NextFrame() override;

  void SkipNextFrame() override;

  void ChangeResolution(size_t width, size_t height) override;

  Resolution GetResolution() const override;

  std::optional<int> fps() const override { return fps_; }

 private:
  YuvFrameReaderImpl::RepeatMode ToYuvFrameReaderRepeatMode(
      RepeatMode repeat_mode) const;
  std::unique_ptr<webrtc::test::FrameReader> frame_reader_ = nullptr;
  std::string filename_;
  size_t width_;
  size_t height_;
  int fps_;
  const RepeatMode repeat_mode_;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_TESTSUPPORT_Y4M_FRAME_GENERATOR_H_
