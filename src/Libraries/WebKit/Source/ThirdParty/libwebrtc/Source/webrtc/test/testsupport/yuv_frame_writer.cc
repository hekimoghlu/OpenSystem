/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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
#include <stdio.h>

#include <string>

#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "test/testsupport/frame_writer.h"

namespace webrtc {
namespace test {

YuvFrameWriterImpl::YuvFrameWriterImpl(std::string output_filename,
                                       int width,
                                       int height)
    : output_filename_(output_filename),
      frame_length_in_bytes_(0),
      width_(width),
      height_(height),
      output_file_(nullptr) {}

YuvFrameWriterImpl::~YuvFrameWriterImpl() {
  Close();
}

bool YuvFrameWriterImpl::Init() {
  if (width_ <= 0 || height_ <= 0) {
    RTC_LOG(LS_ERROR) << "Frame width and height must be positive.";
    return false;
  }
  frame_length_in_bytes_ =
      width_ * height_ + 2 * ((width_ + 1) / 2) * ((height_ + 1) / 2);

  output_file_ = fopen(output_filename_.c_str(), "wb");
  if (output_file_ == nullptr) {
    RTC_LOG(LS_ERROR) << "Couldn't open output file: "
                      << output_filename_.c_str();
    return false;
  }
  return true;
}

bool YuvFrameWriterImpl::WriteFrame(const uint8_t* frame_buffer) {
  RTC_DCHECK(frame_buffer);
  if (output_file_ == nullptr) {
    RTC_LOG(LS_ERROR) << "YuvFrameWriterImpl is not initialized.";
    return false;
  }
  size_t bytes_written =
      fwrite(frame_buffer, 1, frame_length_in_bytes_, output_file_);
  if (bytes_written != frame_length_in_bytes_) {
    RTC_LOG(LS_ERROR) << "Cound't write frame to file: "
                      << output_filename_.c_str();
    return false;
  }
  return true;
}

void YuvFrameWriterImpl::Close() {
  if (output_file_ != nullptr) {
    fclose(output_file_);
    output_file_ = nullptr;
  }
}

size_t YuvFrameWriterImpl::FrameLength() {
  return frame_length_in_bytes_;
}

}  // namespace test
}  // namespace webrtc
