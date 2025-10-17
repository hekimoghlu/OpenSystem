/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 14, 2023.
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
#include "test/configurable_frame_size_encoder.h"

#include <string.h>

#include <cstdint>
#include <type_traits>
#include <utility>

#include "api/video/encoded_image.h"
#include "modules/video_coding/include/video_codec_interface.h"
#include "modules/video_coding/include/video_error_codes.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace test {

ConfigurableFrameSizeEncoder::ConfigurableFrameSizeEncoder(
    size_t max_frame_size)
    : callback_(NULL),
      current_frame_size_(max_frame_size),
      codec_type_(kVideoCodecGeneric) {}

ConfigurableFrameSizeEncoder::~ConfigurableFrameSizeEncoder() {}

void ConfigurableFrameSizeEncoder::SetFecControllerOverride(
    FecControllerOverride* fec_controller_override) {
  // Ignored.
}

int32_t ConfigurableFrameSizeEncoder::InitEncode(
    const VideoCodec* codec_settings,
    const Settings& settings) {
  return WEBRTC_VIDEO_CODEC_OK;
}

int32_t ConfigurableFrameSizeEncoder::Encode(
    const VideoFrame& inputImage,
    const std::vector<VideoFrameType>* frame_types) {
  EncodedImage encodedImage;
  auto buffer = EncodedImageBuffer::Create(current_frame_size_);
  memset(buffer->data(), 0, current_frame_size_);
  encodedImage.SetEncodedData(buffer);
  encodedImage._encodedHeight = inputImage.height();
  encodedImage._encodedWidth = inputImage.width();
  encodedImage._frameType = VideoFrameType::kVideoFrameKey;
  encodedImage.SetRtpTimestamp(inputImage.rtp_timestamp());
  encodedImage.capture_time_ms_ = inputImage.render_time_ms();
  CodecSpecificInfo specific{};
  specific.codecType = codec_type_;
  callback_->OnEncodedImage(encodedImage, &specific);
  if (post_encode_callback_) {
    (*post_encode_callback_)();
  }
  return WEBRTC_VIDEO_CODEC_OK;
}

int32_t ConfigurableFrameSizeEncoder::RegisterEncodeCompleteCallback(
    EncodedImageCallback* callback) {
  callback_ = callback;
  return WEBRTC_VIDEO_CODEC_OK;
}

int32_t ConfigurableFrameSizeEncoder::Release() {
  return WEBRTC_VIDEO_CODEC_OK;
}

void ConfigurableFrameSizeEncoder::SetRates(
    const RateControlParameters& parameters) {}

VideoEncoder::EncoderInfo ConfigurableFrameSizeEncoder::GetEncoderInfo() const {
  return EncoderInfo();
}

int32_t ConfigurableFrameSizeEncoder::SetFrameSize(size_t size) {
  current_frame_size_ = size;
  return WEBRTC_VIDEO_CODEC_OK;
}

void ConfigurableFrameSizeEncoder::SetCodecType(VideoCodecType codec_type) {
  codec_type_ = codec_type;
}

void ConfigurableFrameSizeEncoder::RegisterPostEncodeCallback(
    std::function<void(void)> post_encode_callback) {
  post_encode_callback_ = std::move(post_encode_callback);
}

}  // namespace test
}  // namespace webrtc
