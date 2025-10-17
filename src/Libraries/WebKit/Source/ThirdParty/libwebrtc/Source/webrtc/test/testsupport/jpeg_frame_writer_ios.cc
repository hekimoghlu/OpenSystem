/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 26, 2023.
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
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "test/testsupport/frame_writer.h"

namespace webrtc {
namespace test {

JpegFrameWriter::JpegFrameWriter(const std::string& /*output_filename*/) {}

bool JpegFrameWriter::WriteFrame(const VideoFrame& /*input_frame*/,
                                 int /*quality*/) {
  RTC_LOG(LS_WARNING)
      << "Libjpeg isn't available on IOS. Jpeg frame writer is not "
         "supported. No frame will be saved.";
  // Don't fail.
  return true;
}

}  // namespace test
}  // namespace webrtc
