/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 17, 2025.
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
#include <ostream>

#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/decode_test_driver.h"
#include "test/ivf_video_source.h"
#include "test/util.h"
#include "test/video_source.h"

namespace {

struct ObuExtensionHeader {
  int temporal_id;
  int spatial_id;
};

struct DecodeParam {
  const char *filename;
  const ObuExtensionHeader *headers;
  size_t num_headers;
};

std::ostream &operator<<(std::ostream &os, const DecodeParam &dp) {
  return os << "file: " << dp.filename;
}

class DecodeScalabilityTest
    : public ::libaom_test::DecoderTest,
      public ::libaom_test::CodecTestWithParam<DecodeParam> {
 protected:
  DecodeScalabilityTest()
      : DecoderTest(GET_PARAM(0)), headers_(GET_PARAM(1).headers),
        num_headers_(GET_PARAM(1).num_headers) {}

  ~DecodeScalabilityTest() override = default;

  void PreDecodeFrameHook(const libaom_test::CompressedVideoSource &video,
                          libaom_test::Decoder *decoder) override {
    if (video.frame_number() == 0)
      decoder->Control(AV1D_SET_OUTPUT_ALL_LAYERS, 1);
  }

  void DecompressedFrameHook(const aom_image_t &img,
                             const unsigned int /*frame_number*/) override {
    const ObuExtensionHeader &header = headers_[header_index_];
    EXPECT_EQ(img.temporal_id, header.temporal_id);
    EXPECT_EQ(img.spatial_id, header.spatial_id);
    header_index_ = (header_index_ + 1) % num_headers_;
  }

  void RunTest() {
    const DecodeParam input = GET_PARAM(1);
    aom_codec_dec_cfg_t cfg = { 1, 0, 0, !FORCE_HIGHBITDEPTH_DECODING };
    libaom_test::IVFVideoSource decode_video(input.filename);
    decode_video.Init();

    ASSERT_NO_FATAL_FAILURE(RunLoop(&decode_video, cfg));
  }

 private:
  const ObuExtensionHeader *const headers_;
  const size_t num_headers_;
  size_t header_index_ = 0;
};

TEST_P(DecodeScalabilityTest, ObuExtensionHeader) { RunTest(); }

// For all test files, we have:
//   operatingPoint = 0
//   OperatingPointIdc = operating_point_idc[ 0 ]

// av1-1-b8-01-size-16x16.ivf:
//   operating_points_cnt_minus_1 = 0
//   operating_point_idc[ 0 ] = 0x0
const ObuExtensionHeader kSize16x16Headers[1] = { { 0, 0 } };

// av1-1-b8-22-svc-L1T2.ivf:
//   operating_points_cnt_minus_1 = 1
//   operating_point_idc[ 0 ] = 0x103
//   operating_point_idc[ 1 ] = 0x101
const ObuExtensionHeader kL1T2Headers[2] = { { 0, 0 }, { 1, 0 } };

// av1-1-b8-22-svc-L2T1.ivf:
//   operating_points_cnt_minus_1 = 1
//   operating_point_idc[ 0 ] = 0x301
//   operating_point_idc[ 1 ] = 0x101
const ObuExtensionHeader kL2T1Headers[2] = { { 0, 0 }, { 0, 1 } };

// av1-1-b8-22-svc-L2T2.ivf:
//   operating_points_cnt_minus_1 = 3
//   operating_point_idc[ 0 ] = 0x303
//   operating_point_idc[ 1 ] = 0x301
//   operating_point_idc[ 2 ] = 0x103
//   operating_point_idc[ 3 ] = 0x101
const ObuExtensionHeader kL2T2Headers[4] = {
  { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 }
};

const DecodeParam kAV1DecodeScalabilityTests[] = {
  // { filename, headers, num_headers }
  { "av1-1-b8-01-size-16x16.ivf", kSize16x16Headers, 1 },
  { "av1-1-b8-22-svc-L1T2.ivf", kL1T2Headers, 2 },
  { "av1-1-b8-22-svc-L2T1.ivf", kL2T1Headers, 2 },
  { "av1-1-b8-22-svc-L2T2.ivf", kL2T2Headers, 4 },
};

AV1_INSTANTIATE_TEST_SUITE(DecodeScalabilityTest,
                           ::testing::ValuesIn(kAV1DecodeScalabilityTests));

}  // namespace
