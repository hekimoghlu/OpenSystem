/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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
#include "test/pc/e2e/analyzer/video/simulcast_dummy_buffer_helper.h"

#include "api/video/i420_buffer.h"
#include "api/video/video_frame.h"
#include "api/video/video_frame_buffer.h"

namespace webrtc {
namespace webrtc_pc_e2e {
namespace {

constexpr char kIrrelatedSimulcastStreamFrameData[] = "Dummy!";

}  // namespace

rtc::scoped_refptr<webrtc::VideoFrameBuffer> CreateDummyFrameBuffer() {
  // Use i420 buffer here as default one and supported by all codecs.
  rtc::scoped_refptr<webrtc::I420Buffer> buffer =
      webrtc::I420Buffer::Create(2, 2);
  memcpy(buffer->MutableDataY(), kIrrelatedSimulcastStreamFrameData, 2);
  memcpy(buffer->MutableDataY() + buffer->StrideY(),
         kIrrelatedSimulcastStreamFrameData + 2, 2);
  memcpy(buffer->MutableDataU(), kIrrelatedSimulcastStreamFrameData + 4, 1);
  memcpy(buffer->MutableDataV(), kIrrelatedSimulcastStreamFrameData + 5, 1);
  return buffer;
}

bool IsDummyFrame(const webrtc::VideoFrame& video_frame) {
  if (video_frame.width() != 2 || video_frame.height() != 2) {
    return false;
  }
  rtc::scoped_refptr<webrtc::I420BufferInterface> buffer =
      video_frame.video_frame_buffer()->ToI420();
  if (memcmp(buffer->DataY(), kIrrelatedSimulcastStreamFrameData, 2) != 0) {
    return false;
  }
  if (memcmp(buffer->DataY() + buffer->StrideY(),
             kIrrelatedSimulcastStreamFrameData + 2, 2) != 0) {
    return false;
  }
  if (memcmp(buffer->DataU(), kIrrelatedSimulcastStreamFrameData + 4, 1) != 0) {
    return false;
  }
  if (memcmp(buffer->DataV(), kIrrelatedSimulcastStreamFrameData + 5, 1) != 0) {
    return false;
  }
  return true;
}

}  // namespace webrtc_pc_e2e
}  // namespace webrtc
