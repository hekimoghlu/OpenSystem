/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 15, 2025.
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
#include "test/mappable_native_buffer.h"

#include "absl/algorithm/container.h"
#include "api/video/i420_buffer.h"
#include "api/video/nv12_buffer.h"
#include "api/video/video_frame.h"
#include "api/video/video_rotation.h"
#include "common_video/include/video_frame_buffer.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace test {

namespace {

class NV12BufferWithDidConvertToI420 : public NV12Buffer {
 public:
  NV12BufferWithDidConvertToI420(int width, int height)
      : NV12Buffer(width, height), did_convert_to_i420_(false) {}

  bool did_convert_to_i420() const { return did_convert_to_i420_; }

  rtc::scoped_refptr<I420BufferInterface> ToI420() override {
    did_convert_to_i420_ = true;
    return NV12Buffer::ToI420();
  }

 private:
  bool did_convert_to_i420_;
};

}  // namespace

VideoFrame CreateMappableNativeFrame(int64_t ntp_time_ms,
                                     VideoFrameBuffer::Type mappable_type,
                                     int width,
                                     int height) {
  VideoFrame frame =
      VideoFrame::Builder()
          .set_video_frame_buffer(rtc::make_ref_counted<MappableNativeBuffer>(
              mappable_type, width, height))
          .set_rtp_timestamp(99)
          .set_timestamp_ms(99)
          .set_rotation(kVideoRotation_0)
          .build();
  frame.set_ntp_time_ms(ntp_time_ms);
  return frame;
}

rtc::scoped_refptr<MappableNativeBuffer> GetMappableNativeBufferFromVideoFrame(
    const VideoFrame& frame) {
  return rtc::scoped_refptr<MappableNativeBuffer>(
      static_cast<MappableNativeBuffer*>(frame.video_frame_buffer().get()));
}

MappableNativeBuffer::ScaledBuffer::ScaledBuffer(
    rtc::scoped_refptr<MappableNativeBuffer> parent,
    int width,
    int height)
    : parent_(std::move(parent)), width_(width), height_(height) {}

MappableNativeBuffer::ScaledBuffer::~ScaledBuffer() {}

rtc::scoped_refptr<VideoFrameBuffer>
MappableNativeBuffer::ScaledBuffer::CropAndScale(int offset_x,
                                                 int offset_y,
                                                 int crop_width,
                                                 int crop_height,
                                                 int scaled_width,
                                                 int scaled_height) {
  return rtc::make_ref_counted<ScaledBuffer>(parent_, scaled_width,
                                             scaled_height);
}

rtc::scoped_refptr<I420BufferInterface>
MappableNativeBuffer::ScaledBuffer::ToI420() {
  return parent_->GetOrCreateMappedBuffer(width_, height_)->ToI420();
}

rtc::scoped_refptr<VideoFrameBuffer>
MappableNativeBuffer::ScaledBuffer::GetMappedFrameBuffer(
    rtc::ArrayView<VideoFrameBuffer::Type> types) {
  if (absl::c_find(types, parent_->mappable_type_) == types.end())
    return nullptr;
  return parent_->GetOrCreateMappedBuffer(width_, height_);
}

MappableNativeBuffer::MappableNativeBuffer(VideoFrameBuffer::Type mappable_type,
                                           int width,
                                           int height)
    : mappable_type_(mappable_type), width_(width), height_(height) {
  RTC_DCHECK(mappable_type_ == VideoFrameBuffer::Type::kI420 ||
             mappable_type_ == VideoFrameBuffer::Type::kNV12);
}

MappableNativeBuffer::~MappableNativeBuffer() {}

rtc::scoped_refptr<VideoFrameBuffer> MappableNativeBuffer::CropAndScale(
    int offset_x,
    int offset_y,
    int crop_width,
    int crop_height,
    int scaled_width,
    int scaled_height) {
  return FullSizeBuffer()->CropAndScale(
      offset_x, offset_y, crop_width, crop_height, scaled_width, scaled_height);
}

rtc::scoped_refptr<I420BufferInterface> MappableNativeBuffer::ToI420() {
  return FullSizeBuffer()->ToI420();
}

rtc::scoped_refptr<VideoFrameBuffer> MappableNativeBuffer::GetMappedFrameBuffer(
    rtc::ArrayView<VideoFrameBuffer::Type> types) {
  return FullSizeBuffer()->GetMappedFrameBuffer(types);
}

std::vector<rtc::scoped_refptr<VideoFrameBuffer>>
MappableNativeBuffer::GetMappedFramedBuffers() const {
  MutexLock lock(&lock_);
  return mapped_buffers_;
}

bool MappableNativeBuffer::DidConvertToI420() const {
  if (mappable_type_ != VideoFrameBuffer::Type::kNV12)
    return false;
  MutexLock lock(&lock_);
  for (auto& mapped_buffer : mapped_buffers_) {
    if (static_cast<NV12BufferWithDidConvertToI420*>(mapped_buffer.get())
            ->did_convert_to_i420()) {
      return true;
    }
  }
  return false;
}

rtc::scoped_refptr<MappableNativeBuffer::ScaledBuffer>
MappableNativeBuffer::FullSizeBuffer() {
  return rtc::make_ref_counted<ScaledBuffer>(
      rtc::scoped_refptr<MappableNativeBuffer>(this), width_, height_);
}

rtc::scoped_refptr<VideoFrameBuffer>
MappableNativeBuffer::GetOrCreateMappedBuffer(int width, int height) {
  MutexLock lock(&lock_);
  for (auto& mapped_buffer : mapped_buffers_) {
    if (mapped_buffer->width() == width && mapped_buffer->height() == height) {
      return mapped_buffer;
    }
  }
  rtc::scoped_refptr<VideoFrameBuffer> mapped_buffer;
  switch (mappable_type_) {
    case VideoFrameBuffer::Type::kI420: {
      rtc::scoped_refptr<I420Buffer> i420_buffer =
          I420Buffer::Create(width, height);
      I420Buffer::SetBlack(i420_buffer.get());
      mapped_buffer = i420_buffer;
      break;
    }
    case VideoFrameBuffer::Type::kNV12: {
      auto nv12_buffer =
          rtc::make_ref_counted<NV12BufferWithDidConvertToI420>(width, height);
      nv12_buffer->InitializeData();
      mapped_buffer = std::move(nv12_buffer);
      break;
    }
    default:
      RTC_DCHECK_NOTREACHED();
  }
  mapped_buffers_.push_back(mapped_buffer);
  return mapped_buffer;
}

}  // namespace test
}  // namespace webrtc
