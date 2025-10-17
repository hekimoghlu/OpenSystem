/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 26, 2022.
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
#include "media/base/fake_video_renderer.h"

namespace cricket {
namespace {
bool CheckFrameColorYuv(const webrtc::VideoFrame& frame) {
  // TODO(zhurunz) Check with VP8 team to see if we can remove this
  // tolerance on Y values. Some unit tests produce Y values close
  // to 16 rather than close to zero, for supposedly black frames.
  // Largest value observed is 34, e.g., running
  // PeerConnectionIntegrationTest.SendAndReceive16To9AspectRatio.
  static constexpr uint8_t y_min = 0;
  static constexpr uint8_t y_max = 48;
  static constexpr uint8_t u_min = 128;
  static constexpr uint8_t u_max = 128;
  static constexpr uint8_t v_min = 128;
  static constexpr uint8_t v_max = 128;

  if (!frame.video_frame_buffer()) {
    return false;
  }
  rtc::scoped_refptr<const webrtc::I420BufferInterface> i420_buffer =
      frame.video_frame_buffer()->ToI420();
  // Y
  int y_width = frame.width();
  int y_height = frame.height();
  const uint8_t* y_plane = i420_buffer->DataY();
  const uint8_t* y_pos = y_plane;
  int32_t y_pitch = i420_buffer->StrideY();
  for (int i = 0; i < y_height; ++i) {
    for (int j = 0; j < y_width; ++j) {
      uint8_t y_value = *(y_pos + j);
      if (y_value < y_min || y_value > y_max) {
        return false;
      }
    }
    y_pos += y_pitch;
  }
  // U and V
  int chroma_width = i420_buffer->ChromaWidth();
  int chroma_height = i420_buffer->ChromaHeight();
  const uint8_t* u_plane = i420_buffer->DataU();
  const uint8_t* v_plane = i420_buffer->DataV();
  const uint8_t* u_pos = u_plane;
  const uint8_t* v_pos = v_plane;
  int32_t u_pitch = i420_buffer->StrideU();
  int32_t v_pitch = i420_buffer->StrideV();
  for (int i = 0; i < chroma_height; ++i) {
    for (int j = 0; j < chroma_width; ++j) {
      uint8_t u_value = *(u_pos + j);
      if (u_value < u_min || u_value > u_max) {
        return false;
      }
      uint8_t v_value = *(v_pos + j);
      if (v_value < v_min || v_value > v_max) {
        return false;
      }
    }
    u_pos += u_pitch;
    v_pos += v_pitch;
  }
  return true;
}
}  // namespace

FakeVideoRenderer::FakeVideoRenderer() = default;

void FakeVideoRenderer::OnFrame(const webrtc::VideoFrame& frame) {
  webrtc::MutexLock lock(&mutex_);
  black_frame_ = CheckFrameColorYuv(frame);
  ++num_rendered_frames_;
  width_ = frame.width();
  height_ = frame.height();
  rotation_ = frame.rotation();
  timestamp_us_ = frame.timestamp_us();
}

}  // namespace cricket
