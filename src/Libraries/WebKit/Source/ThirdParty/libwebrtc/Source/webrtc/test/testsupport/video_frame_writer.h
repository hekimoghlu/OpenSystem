/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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
#ifndef TEST_TESTSUPPORT_VIDEO_FRAME_WRITER_H_
#define TEST_TESTSUPPORT_VIDEO_FRAME_WRITER_H_

#include <memory>
#include <string>

#include "api/test/video/video_frame_writer.h"
#include "api/video/video_frame.h"
#include "rtc_base/buffer.h"
#include "test/testsupport/frame_writer.h"

namespace webrtc {
namespace test {

// Writes webrtc::VideoFrame to specified file with y4m frame writer
class Y4mVideoFrameWriterImpl : public VideoFrameWriter {
 public:
  Y4mVideoFrameWriterImpl(std::string output_file_name,
                          int width,
                          int height,
                          int fps);
  ~Y4mVideoFrameWriterImpl() override = default;

  bool WriteFrame(const webrtc::VideoFrame& frame) override;
  void Close() override;

 private:
  const int width_;
  const int height_;

  std::unique_ptr<FrameWriter> frame_writer_;
};

// Writes webrtc::VideoFrame to specified file with yuv frame writer
class YuvVideoFrameWriterImpl : public VideoFrameWriter {
 public:
  YuvVideoFrameWriterImpl(std::string output_file_name, int width, int height);
  ~YuvVideoFrameWriterImpl() override = default;

  bool WriteFrame(const webrtc::VideoFrame& frame) override;
  void Close() override;

 private:
  const int width_;
  const int height_;

  std::unique_ptr<FrameWriter> frame_writer_;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_TESTSUPPORT_VIDEO_FRAME_WRITER_H_
