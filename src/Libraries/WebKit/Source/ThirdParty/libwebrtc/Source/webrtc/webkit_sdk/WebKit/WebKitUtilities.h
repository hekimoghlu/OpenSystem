/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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
#pragma once

#include "api/video/video_frame_buffer.h"
#include "api/video/video_rotation.h"
#include "api/scoped_refptr.h"
#include <CoreFoundation/CFBase.h>
#include <functional>

using CVPixelBufferRef = struct __CVBuffer*;

namespace webrtc {

class VideoFrame;

enum class WebKitH265 { Off, On };
enum class WebKitVP9 { Off, Profile0, Profile0And2 };
enum class WebKitVP9VTB { Off, On };
enum class WebKitAv1 { Off, On };

void setH264HardwareEncoderAllowed(bool);
bool isH264HardwareEncoderAllowed();

enum class BufferType { I420, I010, I422, I210 };
CVPixelBufferRef copyPixelBufferForFrame(const VideoFrame&) CF_RETURNS_RETAINED;
CVPixelBufferRef createPixelBufferFromFrame(const VideoFrame&, const std::function<CVPixelBufferRef(size_t, size_t, BufferType)>& createPixelBuffer) CF_RETURNS_RETAINED;
CVPixelBufferRef createPixelBufferFromFrameBuffer(VideoFrameBuffer&, const std::function<CVPixelBufferRef(size_t, size_t, BufferType)>& createPixelBuffer) CF_RETURNS_RETAINED;
rtc::scoped_refptr<webrtc::VideoFrameBuffer> pixelBufferToFrame(CVPixelBufferRef);
bool copyVideoFrameBuffer(VideoFrameBuffer&, uint8_t*);

typedef CVPixelBufferRef (*GetBufferCallback)(void*);
typedef void (*ReleaseBufferCallback)(void*);
rtc::scoped_refptr<VideoFrameBuffer> toWebRTCVideoFrameBuffer(void*, GetBufferCallback, ReleaseBufferCallback, int width, int height);
void* videoFrameBufferProvider(const VideoFrame&);

bool convertBGRAToYUV(CVPixelBufferRef sourceBuffer, CVPixelBufferRef destinationBuffer);

struct I420BufferLayout {
    size_t offsetY { 0 };
    size_t strideY { 0 };
    size_t offsetU { 0 };
    size_t strideU { 0 };
    size_t offsetV { 0 };
    size_t strideV { 0 };
};

struct I420ABufferLayout : I420BufferLayout {
    size_t offsetA { 0 };
    size_t strideA { 0 };
};

CVPixelBufferRef createPixelBufferFromI420Buffer(const uint8_t* buffer, size_t length, size_t width, size_t height, I420BufferLayout) CF_RETURNS_RETAINED;
CVPixelBufferRef createPixelBufferFromI420ABuffer(const uint8_t* buffer, size_t length, size_t width, size_t height, I420ABufferLayout) CF_RETURNS_RETAINED;

}
