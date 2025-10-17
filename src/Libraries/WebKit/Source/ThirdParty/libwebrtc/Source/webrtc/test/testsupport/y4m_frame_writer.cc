/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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

#include "rtc_base/logging.h"
#include "test/testsupport/frame_writer.h"

namespace webrtc {
namespace test {

Y4mFrameWriterImpl::Y4mFrameWriterImpl(std::string output_filename,
                                       int width,
                                       int height,
                                       int frame_rate)
    : YuvFrameWriterImpl(output_filename, width, height),
      frame_rate_(frame_rate) {}

Y4mFrameWriterImpl::~Y4mFrameWriterImpl() = default;

bool Y4mFrameWriterImpl::Init() {
  if (!YuvFrameWriterImpl::Init()) {
    return false;
  }
  int bytes_written = fprintf(output_file_, "YUV4MPEG2 W%d H%d F%d:1 C420\n",
                              width_, height_, frame_rate_);
  if (bytes_written < 0) {
    RTC_LOG(LS_ERROR) << "Failed to write Y4M file header to file: "
                      << output_filename_.c_str();
    return false;
  }
  return true;
}

bool Y4mFrameWriterImpl::WriteFrame(const uint8_t* frame_buffer) {
  if (output_file_ == nullptr) {
    RTC_LOG(LS_ERROR) << "Y4mFrameWriterImpl is not initialized.";
    return false;
  }
  int bytes_written = fprintf(output_file_, "FRAME\n");
  if (bytes_written < 0) {
    RTC_LOG(LS_ERROR) << "Couldn't write Y4M frame header to file: "
                      << output_filename_.c_str();
    return false;
  }
  return YuvFrameWriterImpl::WriteFrame(frame_buffer);
}

}  // namespace test
}  // namespace webrtc
