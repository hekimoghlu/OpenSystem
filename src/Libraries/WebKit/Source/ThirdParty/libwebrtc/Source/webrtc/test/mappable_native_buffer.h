/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 28, 2025.
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
#ifndef TEST_MAPPABLE_NATIVE_BUFFER_H_
#define TEST_MAPPABLE_NATIVE_BUFFER_H_

#include <utility>
#include <vector>

#include "api/array_view.h"
#include "api/video/video_frame.h"
#include "common_video/include/video_frame_buffer.h"
#include "rtc_base/ref_counted_object.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {
namespace test {

class MappableNativeBuffer;

VideoFrame CreateMappableNativeFrame(int64_t ntp_time_ms,
                                     VideoFrameBuffer::Type mappable_type,
                                     int width,
                                     int height);

rtc::scoped_refptr<MappableNativeBuffer> GetMappableNativeBufferFromVideoFrame(
    const VideoFrame& frame);

// A for-testing native buffer that is scalable and mappable. The contents of
// the buffer is black and the pixels are created upon mapping. Mapped buffers
// are stored inside MappableNativeBuffer, allowing tests to verify which
// resolutions were mapped, e.g. when passing them in to an encoder or other
// modules.
class MappableNativeBuffer : public VideoFrameBuffer {
 public:
  // If `allow_i420_conversion` is false, calling ToI420() on a non-I420 buffer
  // will DCHECK-crash. Used to ensure zero-copy in tests.
  MappableNativeBuffer(VideoFrameBuffer::Type mappable_type,
                       int width,
                       int height);
  ~MappableNativeBuffer() override;

  VideoFrameBuffer::Type mappable_type() const { return mappable_type_; }

  VideoFrameBuffer::Type type() const override { return Type::kNative; }
  int width() const override { return width_; }
  int height() const override { return height_; }

  rtc::scoped_refptr<VideoFrameBuffer> CropAndScale(int offset_x,
                                                    int offset_y,
                                                    int crop_width,
                                                    int crop_height,
                                                    int scaled_width,
                                                    int scaled_height) override;

  rtc::scoped_refptr<I420BufferInterface> ToI420() override;
  rtc::scoped_refptr<VideoFrameBuffer> GetMappedFrameBuffer(
      rtc::ArrayView<VideoFrameBuffer::Type> types) override;

  // Gets all the buffers that have been mapped so far, including mappings of
  // cropped and scaled buffers.
  std::vector<rtc::scoped_refptr<VideoFrameBuffer>> GetMappedFramedBuffers()
      const;
  bool DidConvertToI420() const;

 private:
  friend class RefCountedObject<MappableNativeBuffer>;

  class ScaledBuffer : public VideoFrameBuffer {
   public:
    ScaledBuffer(rtc::scoped_refptr<MappableNativeBuffer> parent,
                 int width,
                 int height);
    ~ScaledBuffer() override;

    VideoFrameBuffer::Type type() const override { return Type::kNative; }
    int width() const override { return width_; }
    int height() const override { return height_; }

    rtc::scoped_refptr<VideoFrameBuffer> CropAndScale(
        int offset_x,
        int offset_y,
        int crop_width,
        int crop_height,
        int scaled_width,
        int scaled_height) override;

    rtc::scoped_refptr<I420BufferInterface> ToI420() override;
    rtc::scoped_refptr<VideoFrameBuffer> GetMappedFrameBuffer(
        rtc::ArrayView<VideoFrameBuffer::Type> types) override;

   private:
    friend class RefCountedObject<ScaledBuffer>;

    const rtc::scoped_refptr<MappableNativeBuffer> parent_;
    const int width_;
    const int height_;
  };

  rtc::scoped_refptr<ScaledBuffer> FullSizeBuffer();
  rtc::scoped_refptr<VideoFrameBuffer> GetOrCreateMappedBuffer(int width,
                                                               int height);

  const VideoFrameBuffer::Type mappable_type_;
  const int width_;
  const int height_;
  mutable Mutex lock_;
  std::vector<rtc::scoped_refptr<VideoFrameBuffer>> mapped_buffers_
      RTC_GUARDED_BY(&lock_);
};

}  // namespace test
}  // namespace webrtc

#endif  //  TEST_MAPPABLE_NATIVE_BUFFER_H_
