/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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
#include "modules/video_coding/codecs/test/encoded_video_frame_producer.h"

#include <memory>
#include <vector>

#include "api/test/create_frame_generator.h"
#include "api/test/frame_generator_interface.h"
#include "api/transport/rtp/dependency_descriptor.h"
#include "api/video/video_frame.h"
#include "api/video/video_frame_type.h"
#include "api/video_codecs/video_encoder.h"
#include "modules/video_coding/include/video_codec_interface.h"
#include "modules/video_coding/include/video_error_codes.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace {

class EncoderCallback : public EncodedImageCallback {
 public:
  explicit EncoderCallback(
      std::vector<EncodedVideoFrameProducer::EncodedFrame>& output_frames)
      : output_frames_(output_frames) {}

 private:
  Result OnEncodedImage(const EncodedImage& encoded_image,
                        const CodecSpecificInfo* codec_specific_info) override {
    output_frames_.push_back({encoded_image, *codec_specific_info});
    return Result(Result::Error::OK);
  }

  std::vector<EncodedVideoFrameProducer::EncodedFrame>& output_frames_;
};

}  // namespace

std::vector<EncodedVideoFrameProducer::EncodedFrame>
EncodedVideoFrameProducer::Encode() {
  std::unique_ptr<test::FrameGeneratorInterface> frame_buffer_generator =
      test::CreateSquareFrameGenerator(
          resolution_.Width(), resolution_.Height(),
          test::FrameGeneratorInterface::OutputType::kI420, std::nullopt);

  std::vector<EncodedFrame> encoded_frames;
  EncoderCallback encoder_callback(encoded_frames);
  RTC_CHECK_EQ(encoder_.RegisterEncodeCompleteCallback(&encoder_callback),
               WEBRTC_VIDEO_CODEC_OK);

  uint32_t rtp_tick = 90000 / framerate_fps_;
  for (int i = 0; i < num_input_frames_; ++i) {
    VideoFrame frame =
        VideoFrame::Builder()
            .set_video_frame_buffer(frame_buffer_generator->NextFrame().buffer)
            .set_rtp_timestamp(rtp_timestamp_)
            .set_presentation_timestamp(presentation_timestamp_)
            .build();
    rtp_timestamp_ += rtp_tick;
    RTC_CHECK_EQ(encoder_.Encode(frame, &next_frame_type_),
                 WEBRTC_VIDEO_CODEC_OK);
    next_frame_type_[0] = VideoFrameType::kVideoFrameDelta;
  }

  RTC_CHECK_EQ(encoder_.RegisterEncodeCompleteCallback(nullptr),
               WEBRTC_VIDEO_CODEC_OK);
  return encoded_frames;
}

}  // namespace webrtc
