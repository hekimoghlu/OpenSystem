/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
#include "webkit_sdk/objc/native/src/objc_frame_buffer.h"

#import "base/RTCVideoFrameBuffer.h"
#import "api/make_ref_counted.h"
#import "webkit_sdk/objc/api/video_frame_buffer/RTCNativeI420Buffer+Private.h"
#import "webkit_sdk/objc/components/video_frame_buffer/RTCCVPixelBuffer.h"

namespace webrtc {

namespace {

/** ObjCFrameBuffer that conforms to I420BufferInterface by wrapping RTCI420Buffer */
class ObjCI420FrameBuffer : public I420BufferInterface {
 public:
  explicit ObjCI420FrameBuffer(id<RTCI420Buffer> frame_buffer)
      : frame_buffer_(frame_buffer), width_(frame_buffer.width), height_(frame_buffer.height) {}
  ~ObjCI420FrameBuffer() override {
#if defined(WEBRTC_WEBKIT_BUILD)
    [frame_buffer_ close];
#endif
  }

  int width() const override { return width_; }

  int height() const override { return height_; }

  const uint8_t* DataY() const override { return frame_buffer_.dataY; }

  const uint8_t* DataU() const override { return frame_buffer_.dataU; }

  const uint8_t* DataV() const override { return frame_buffer_.dataV; }

  int StrideY() const override { return frame_buffer_.strideY; }

  int StrideU() const override { return frame_buffer_.strideU; }

  int StrideV() const override { return frame_buffer_.strideV; }

 private:
  id<RTCI420Buffer> frame_buffer_;
  int width_;
  int height_;
};

}  // namespace

ObjCFrameBuffer::ObjCFrameBuffer(id<RTCVideoFrameBuffer> frame_buffer)
    : frame_buffer_(frame_buffer), width_(frame_buffer.width), height_(frame_buffer.height) {}

ObjCFrameBuffer::ObjCFrameBuffer(BufferProvider provider, int width, int height)
    : frame_buffer_provider_(provider), width_(width), height_(height) { }

ObjCFrameBuffer::ObjCFrameBuffer(id<RTCVideoFrameBuffer> frame_buffer, int width, int height)
    : frame_buffer_(frame_buffer), width_(width), height_(height) { }

ObjCFrameBuffer::~ObjCFrameBuffer() {
  if (!original_frame_) {
    [frame_buffer_ close];
    if (frame_buffer_provider_.releaseBuffer)
      frame_buffer_provider_.releaseBuffer(frame_buffer_provider_.pointer);
  }
}

VideoFrameBuffer::Type ObjCFrameBuffer::type() const {
  return Type::kNative;
}

int ObjCFrameBuffer::width() const {
  return width_;
}

int ObjCFrameBuffer::height() const {
  return height_;
}

rtc::scoped_refptr<I420BufferInterface> ObjCFrameBuffer::ToI420() {
  auto frameBuffer = wrapped_frame_buffer();
  if (!frameBuffer)
    return nullptr;
  rtc::scoped_refptr<I420BufferInterface> buffer =
    rtc::make_ref_counted<ObjCI420FrameBuffer>([wrapped_frame_buffer() toI420]);

  if (width_ != buffer->width() || height_ != buffer->height())
      buffer = buffer->Scale(width_, height_)->ToI420();

  return buffer;
}

id<RTCVideoFrameBuffer> ObjCFrameBuffer::wrapped_frame_buffer() const {
#if defined(WEBRTC_WEBKIT_BUILD)
  webrtc::MutexLock lock(&mutex_);
  if (!frame_buffer_ && frame_buffer_provider_.getBuffer && frame_buffer_provider_.pointer) {
    if (auto* buffer = frame_buffer_provider_.getBuffer(frame_buffer_provider_.pointer))
      const_cast<ObjCFrameBuffer*>(this)->frame_buffer_ = [[RTCCVPixelBuffer alloc] initWithPixelBuffer:frame_buffer_provider_.getBuffer(frame_buffer_provider_.pointer)];
  }
#endif
  return frame_buffer_;
}

rtc::scoped_refptr<VideoFrameBuffer> ObjCFrameBuffer::CropAndScale(int offset_x, int offset_y, int crop_width, int crop_height, int scaled_width, int scaled_height)
{
  if (offset_x || offset_y || crop_width != width() || crop_height != height())
    return VideoFrameBuffer::CropAndScale(offset_x, offset_y, crop_width, crop_height, scaled_width, scaled_height);

  webrtc::MutexLock lock(&mutex_);
  auto newFrame = (!frame_buffer_ && frame_buffer_provider_.getBuffer && frame_buffer_provider_.pointer) ?
    rtc::make_ref_counted<ObjCFrameBuffer>(frame_buffer_provider_, scaled_width, scaled_height) :
    rtc::make_ref_counted<ObjCFrameBuffer>(frame_buffer_, scaled_width, scaled_height);

  newFrame->set_original_frame(original_frame_ ? *original_frame_ : *this);
  return newFrame;
}

id<RTCVideoFrameBuffer> ToObjCVideoFrameBuffer(const rtc::scoped_refptr<VideoFrameBuffer>& buffer) {
  if (buffer->type() == VideoFrameBuffer::Type::kNative) {
    return static_cast<ObjCFrameBuffer*>(buffer.get())->wrapped_frame_buffer();
  } else {
    return [[RTCI420Buffer alloc] initWithFrameBuffer:buffer->ToI420()];
  }
}

}  // namespace webrtc
