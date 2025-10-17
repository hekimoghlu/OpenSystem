/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 4, 2023.
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
#include <stdint.h>

#include <memory>
#include <optional>

#include "api/video/color_space.h"
#include "api/video/encoded_image.h"
#include "api/video/video_frame.h"
#include "api/video_codecs/video_codec.h"
#include "api/video_codecs/video_decoder.h"
#include "api/video_codecs/video_encoder.h"
#include "common_video/libyuv/include/webrtc_libyuv.h"
#include "media/base/codec.h"
#include "media/base/media_constants.h"
#include "modules/video_coding/codecs/h264/include/h264.h"
#include "modules/video_coding/codecs/test/video_codec_unittest.h"
#include "modules/video_coding/include/video_codec_interface.h"
#include "modules/video_coding/include/video_error_codes.h"
#include "test/gtest.h"
#include "test/video_codec_settings.h"

namespace webrtc {

class TestH264Impl : public VideoCodecUnitTest {
 protected:
  std::unique_ptr<VideoEncoder> CreateEncoder() override {
    return CreateH264Encoder(env_);
  }

  std::unique_ptr<VideoDecoder> CreateDecoder() override {
    return H264Decoder::Create();
  }

  void ModifyCodecSettings(VideoCodec* codec_settings) override {
    webrtc::test::CodecSettings(kVideoCodecH264, codec_settings);
  }
};

#ifdef WEBRTC_USE_H264
#define MAYBE_EncodeDecode EncodeDecode
#define MAYBE_DecodedQpEqualsEncodedQp DecodedQpEqualsEncodedQp
#else
#define MAYBE_EncodeDecode DISABLED_EncodeDecode
#define MAYBE_DecodedQpEqualsEncodedQp DISABLED_DecodedQpEqualsEncodedQp
#endif

TEST_F(TestH264Impl, MAYBE_EncodeDecode) {
  VideoFrame input_frame = NextInputFrame();
  EXPECT_EQ(WEBRTC_VIDEO_CODEC_OK, encoder_->Encode(input_frame, nullptr));
  EncodedImage encoded_frame;
  CodecSpecificInfo codec_specific_info;
  ASSERT_TRUE(WaitForEncodedFrame(&encoded_frame, &codec_specific_info));
  // First frame should be a key frame.
  encoded_frame._frameType = VideoFrameType::kVideoFrameKey;
  EXPECT_EQ(WEBRTC_VIDEO_CODEC_OK, decoder_->Decode(encoded_frame, 0));
  std::unique_ptr<VideoFrame> decoded_frame;
  std::optional<uint8_t> decoded_qp;
  ASSERT_TRUE(WaitForDecodedFrame(&decoded_frame, &decoded_qp));
  ASSERT_TRUE(decoded_frame);
  EXPECT_GT(I420PSNR(&input_frame, decoded_frame.get()), 36);

  const ColorSpace color_space = *decoded_frame->color_space();
  EXPECT_EQ(ColorSpace::PrimaryID::kUnspecified, color_space.primaries());
  EXPECT_EQ(ColorSpace::TransferID::kUnspecified, color_space.transfer());
  EXPECT_EQ(ColorSpace::MatrixID::kUnspecified, color_space.matrix());
  EXPECT_EQ(ColorSpace::RangeID::kInvalid, color_space.range());
  EXPECT_EQ(ColorSpace::ChromaSiting::kUnspecified,
            color_space.chroma_siting_horizontal());
  EXPECT_EQ(ColorSpace::ChromaSiting::kUnspecified,
            color_space.chroma_siting_vertical());
}

TEST_F(TestH264Impl, MAYBE_DecodedQpEqualsEncodedQp) {
  EXPECT_EQ(WEBRTC_VIDEO_CODEC_OK, encoder_->Encode(NextInputFrame(), nullptr));
  EncodedImage encoded_frame;
  CodecSpecificInfo codec_specific_info;
  ASSERT_TRUE(WaitForEncodedFrame(&encoded_frame, &codec_specific_info));
  // First frame should be a key frame.
  encoded_frame._frameType = VideoFrameType::kVideoFrameKey;
  EXPECT_EQ(WEBRTC_VIDEO_CODEC_OK, decoder_->Decode(encoded_frame, 0));
  std::unique_ptr<VideoFrame> decoded_frame;
  std::optional<uint8_t> decoded_qp;
  ASSERT_TRUE(WaitForDecodedFrame(&decoded_frame, &decoded_qp));
  ASSERT_TRUE(decoded_frame);
  ASSERT_TRUE(decoded_qp);
  EXPECT_EQ(encoded_frame.qp_, *decoded_qp);
}

}  // namespace webrtc
