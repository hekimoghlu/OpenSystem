/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 21, 2023.
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
#ifndef SDK_OBJC_NATIVE_SRC_OBJC_FRAME_BUFFER_H_
#define SDK_OBJC_NATIVE_SRC_OBJC_FRAME_BUFFER_H_

#import <CoreVideo/CoreVideo.h>

#include "common_video/include/video_frame_buffer.h"
#include "rtc_base/synchronization/mutex.h"

@protocol RTCVideoFrameBuffer;

namespace webrtc {

typedef CVPixelBufferRef (*GetBufferCallback)(void*);
typedef void (*ReleaseBufferCallback)(void*);

class ObjCFrameBuffer : public VideoFrameBuffer {
 public:
  explicit ObjCFrameBuffer(id<RTCVideoFrameBuffer>);
  ~ObjCFrameBuffer() override;

  struct BufferProvider {
    void *pointer { nullptr };
    GetBufferCallback getBuffer { nullptr };
    ReleaseBufferCallback releaseBuffer { nullptr };
  };
  ObjCFrameBuffer(BufferProvider, int width, int height);
  ObjCFrameBuffer(id<RTCVideoFrameBuffer>, int width, int height);

  Type type() const override;

  int width() const override;
  int height() const override;

  rtc::scoped_refptr<I420BufferInterface> ToI420() override;

  id<RTCVideoFrameBuffer> wrapped_frame_buffer() const;
  void* frame_buffer_provider() { return frame_buffer_provider_.pointer; }

 private:
   rtc::scoped_refptr<VideoFrameBuffer> CropAndScale(int offset_x, int offset_y, int crop_width, int crop_height, int scaled_width, int scaled_height) final;

  void set_original_frame(ObjCFrameBuffer& frame) { original_frame_ = &frame; }

  id<RTCVideoFrameBuffer> frame_buffer_;
  BufferProvider frame_buffer_provider_;
  rtc::scoped_refptr<ObjCFrameBuffer> original_frame_;
  int width_;
  int height_;
  mutable webrtc::Mutex mutex_;
};

id<RTCVideoFrameBuffer> ToObjCVideoFrameBuffer(
    const rtc::scoped_refptr<VideoFrameBuffer>& buffer);

}  // namespace webrtc

#endif  // SDK_OBJC_NATIVE_SRC_OBJC_FRAME_BUFFER_H_
