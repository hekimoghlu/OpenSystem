/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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
#include "test/pc/e2e/analyzer/video/video_dumping.h"

#include <stdio.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/test/video/video_frame_writer.h"
#include "api/video/video_frame.h"
#include "rtc_base/logging.h"
#include "system_wrappers/include/clock.h"
#include "test/testsupport/video_frame_writer.h"

namespace webrtc {
namespace webrtc_pc_e2e {
namespace {

class VideoFrameIdsWriter final : public test::VideoFrameWriter {
 public:
  explicit VideoFrameIdsWriter(absl::string_view file_name)
      : file_name_(file_name) {
    output_file_ = fopen(file_name_.c_str(), "wb");
    RTC_LOG(LS_INFO) << "Writing VideoFrame IDs into " << file_name_;
    RTC_CHECK(output_file_ != nullptr)
        << "Failed to open file to dump frame ids for writing: " << file_name_;
  }
  ~VideoFrameIdsWriter() override { Close(); }

  bool WriteFrame(const VideoFrame& frame) override {
    RTC_CHECK(output_file_ != nullptr) << "Writer is already closed";
    int chars_written = fprintf(output_file_, "%d\n", frame.id());
    if (chars_written < 2) {
      RTC_LOG(LS_ERROR) << "Failed to write frame id to the output file: "
                        << file_name_;
      return false;
    }
    return true;
  }

  void Close() override {
    if (output_file_ != nullptr) {
      RTC_LOG(LS_INFO) << "Closing file for VideoFrame IDs: " << file_name_;
      fclose(output_file_);
      output_file_ = nullptr;
    }
  }

 private:
  const std::string file_name_;
  FILE* output_file_;
};

// Broadcast received frame to multiple underlying frame writers.
class BroadcastingFrameWriter final : public test::VideoFrameWriter {
 public:
  explicit BroadcastingFrameWriter(
      std::vector<std::unique_ptr<test::VideoFrameWriter>> delegates)
      : delegates_(std::move(delegates)) {}
  ~BroadcastingFrameWriter() override { Close(); }

  bool WriteFrame(const webrtc::VideoFrame& frame) override {
    for (auto& delegate : delegates_) {
      if (!delegate->WriteFrame(frame)) {
        return false;
      }
    }
    return true;
  }

  void Close() override {
    for (auto& delegate : delegates_) {
      delegate->Close();
    }
  }

 private:
  std::vector<std::unique_ptr<test::VideoFrameWriter>> delegates_;
};

}  // namespace

VideoWriter::VideoWriter(test::VideoFrameWriter* video_writer,
                         int sampling_modulo)
    : video_writer_(video_writer), sampling_modulo_(sampling_modulo) {}

void VideoWriter::OnFrame(const VideoFrame& frame) {
  if (frames_counter_++ % sampling_modulo_ != 0) {
    return;
  }
  bool result = video_writer_->WriteFrame(frame);
  RTC_CHECK(result) << "Failed to write frame";
}

std::unique_ptr<test::VideoFrameWriter> CreateVideoFrameWithIdsWriter(
    std::unique_ptr<test::VideoFrameWriter> video_writer_delegate,
    absl::string_view frame_ids_dump_file_name) {
  std::vector<std::unique_ptr<test::VideoFrameWriter>> requested_writers;
  requested_writers.push_back(std::move(video_writer_delegate));
  requested_writers.push_back(
      std::make_unique<VideoFrameIdsWriter>(frame_ids_dump_file_name));
  return std::make_unique<BroadcastingFrameWriter>(
      std::move(requested_writers));
}

}  // namespace webrtc_pc_e2e
}  // namespace webrtc
