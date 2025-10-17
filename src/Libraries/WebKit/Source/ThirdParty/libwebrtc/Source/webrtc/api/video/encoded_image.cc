/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 7, 2023.
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
#include "api/video/encoded_image.h"

#include <stdlib.h>

#include <algorithm>
#include <cstdint>
#include <optional>

#include "api/make_ref_counted.h"
#include "api/scoped_refptr.h"
#include "api/units/timestamp.h"
#include "rtc_base/buffer.h"
#include "rtc_base/checks.h"

namespace webrtc {

EncodedImageBuffer::EncodedImageBuffer(size_t size) : buffer_(size) {}

EncodedImageBuffer::EncodedImageBuffer(const uint8_t* data, size_t size)
    : buffer_(data, size) {}

EncodedImageBuffer::EncodedImageBuffer(rtc::Buffer buffer)
    : buffer_(std::move(buffer)) {}

// static
scoped_refptr<EncodedImageBuffer> EncodedImageBuffer::Create(size_t size) {
  return make_ref_counted<EncodedImageBuffer>(size);
}
// static
scoped_refptr<EncodedImageBuffer> EncodedImageBuffer::Create(
    const uint8_t* data,
    size_t size) {
  return make_ref_counted<EncodedImageBuffer>(data, size);
}
// static
scoped_refptr<EncodedImageBuffer> EncodedImageBuffer::Create(
    rtc::Buffer buffer) {
  return make_ref_counted<EncodedImageBuffer>(std::move(buffer));
}

const uint8_t* EncodedImageBuffer::data() const {
  return buffer_.data();
}
uint8_t* EncodedImageBuffer::data() {
  return buffer_.data();
}
size_t EncodedImageBuffer::size() const {
  return buffer_.size();
}

void EncodedImageBuffer::Realloc(size_t size) {
  buffer_.SetSize(size);
}

EncodedImage::EncodedImage() = default;

EncodedImage::EncodedImage(EncodedImage&&) = default;
EncodedImage::EncodedImage(const EncodedImage&) = default;

EncodedImage::~EncodedImage() = default;

EncodedImage& EncodedImage::operator=(EncodedImage&&) = default;
EncodedImage& EncodedImage::operator=(const EncodedImage&) = default;

void EncodedImage::SetEncodeTime(int64_t encode_start_ms,
                                 int64_t encode_finish_ms) {
  timing_.encode_start_ms = encode_start_ms;
  timing_.encode_finish_ms = encode_finish_ms;
}

webrtc::Timestamp EncodedImage::CaptureTime() const {
  return capture_time_ms_ > 0 ? Timestamp::Millis(capture_time_ms_)
                              : Timestamp::MinusInfinity();
}

std::optional<size_t> EncodedImage::SpatialLayerFrameSize(
    int spatial_index) const {
  RTC_DCHECK_GE(spatial_index, 0);
  RTC_DCHECK_LE(spatial_index, spatial_index_.value_or(0));

  auto it = spatial_layer_frame_size_bytes_.find(spatial_index);
  if (it == spatial_layer_frame_size_bytes_.end()) {
    return std::nullopt;
  }

  return it->second;
}

void EncodedImage::SetSpatialLayerFrameSize(int spatial_index,
                                            size_t size_bytes) {
  RTC_DCHECK_GE(spatial_index, 0);
  RTC_DCHECK_LE(spatial_index, spatial_index_.value_or(0));
  RTC_DCHECK_GE(size_bytes, 0);
  spatial_layer_frame_size_bytes_[spatial_index] = size_bytes;
}

}  // namespace webrtc
