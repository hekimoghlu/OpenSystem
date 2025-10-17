/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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
#ifndef COMMON_VIDEO_INCLUDE_VIDEO_FRAME_BUFFER_H_
#define COMMON_VIDEO_INCLUDE_VIDEO_FRAME_BUFFER_H_

#include <stdint.h>

#include <functional>

#include "api/scoped_refptr.h"
#include "api/video/video_frame_buffer.h"

namespace webrtc {

rtc::scoped_refptr<I420BufferInterface> WrapI420Buffer(
    int width,
    int height,
    const uint8_t* y_plane,
    int y_stride,
    const uint8_t* u_plane,
    int u_stride,
    const uint8_t* v_plane,
    int v_stride,
    std::function<void()> no_longer_used);

rtc::scoped_refptr<I422BufferInterface> WrapI422Buffer(
    int width,
    int height,
    const uint8_t* y_plane,
    int y_stride,
    const uint8_t* u_plane,
    int u_stride,
    const uint8_t* v_plane,
    int v_stride,
    std::function<void()> no_longer_used);

rtc::scoped_refptr<I444BufferInterface> WrapI444Buffer(
    int width,
    int height,
    const uint8_t* y_plane,
    int y_stride,
    const uint8_t* u_plane,
    int u_stride,
    const uint8_t* v_plane,
    int v_stride,
    std::function<void()> no_longer_used);

rtc::scoped_refptr<I420ABufferInterface> WrapI420ABuffer(
    int width,
    int height,
    const uint8_t* y_plane,
    int y_stride,
    const uint8_t* u_plane,
    int u_stride,
    const uint8_t* v_plane,
    int v_stride,
    const uint8_t* a_plane,
    int a_stride,
    std::function<void()> no_longer_used);

rtc::scoped_refptr<PlanarYuvBuffer> WrapYuvBuffer(
    VideoFrameBuffer::Type type,
    int width,
    int height,
    const uint8_t* y_plane,
    int y_stride,
    const uint8_t* u_plane,
    int u_stride,
    const uint8_t* v_plane,
    int v_stride,
    std::function<void()> no_longer_used);

rtc::scoped_refptr<I010BufferInterface> WrapI010Buffer(
    int width,
    int height,
    const uint16_t* y_plane,
    int y_stride,
    const uint16_t* u_plane,
    int u_stride,
    const uint16_t* v_plane,
    int v_stride,
    std::function<void()> no_longer_used);

rtc::scoped_refptr<I210BufferInterface> WrapI210Buffer(
    int width,
    int height,
    const uint16_t* y_plane,
    int y_stride,
    const uint16_t* u_plane,
    int u_stride,
    const uint16_t* v_plane,
    int v_stride,
    std::function<void()> no_longer_used);

rtc::scoped_refptr<I410BufferInterface> WrapI410Buffer(
    int width,
    int height,
    const uint16_t* y_plane,
    int y_stride,
    const uint16_t* u_plane,
    int u_stride,
    const uint16_t* v_plane,
    int v_stride,
    std::function<void()> no_longer_used);
}  // namespace webrtc

#endif  // COMMON_VIDEO_INCLUDE_VIDEO_FRAME_BUFFER_H_
