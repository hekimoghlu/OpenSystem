/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 11, 2023.
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
#include <string>

#include "test/codec_factory.h"
#include "test/decode_test_driver.h"
#include "test/md5_helper.h"
#include "test/util.h"
#include "test/webm_video_source.h"

namespace {

const char kVp9TestFile[] = "vp90-2-08-tile_1x8_frame_parallel.webm";
const char kVp9Md5File[] = "vp90-2-08-tile_1x8_frame_parallel.webm.md5";

// Class for testing shutting off the loop filter.
class SkipLoopFilterTest {
 public:
  SkipLoopFilterTest()
      : video_(nullptr), decoder_(nullptr), md5_file_(nullptr) {}

  ~SkipLoopFilterTest() {
    if (md5_file_ != nullptr) fclose(md5_file_);
    delete decoder_;
    delete video_;
  }

  // If |threads| > 0 then set the decoder with that number of threads.
  bool Init(int num_threads) {
    expected_md5_[0] = '\0';
    junk_[0] = '\0';
    video_ = new libvpx_test::WebMVideoSource(kVp9TestFile);
    if (video_ == nullptr) {
      EXPECT_NE(video_, nullptr);
      return false;
    }
    video_->Init();
    video_->Begin();

    vpx_codec_dec_cfg_t cfg = vpx_codec_dec_cfg_t();
    if (num_threads > 0) cfg.threads = num_threads;
    decoder_ = new libvpx_test::VP9Decoder(cfg, 0);
    if (decoder_ == nullptr) {
      EXPECT_NE(decoder_, nullptr);
      return false;
    }

    OpenMd5File(kVp9Md5File);
    return !::testing::Test::HasFailure();
  }

  // Set the VP9 skipLoopFilter control value.
  void SetSkipLoopFilter(int value, vpx_codec_err_t expected_value) {
    ASSERT_NE(decoder_, nullptr);
    decoder_->Control(VP9_SET_SKIP_LOOP_FILTER, value, expected_value);
  }

  vpx_codec_err_t DecodeOneFrame() {
    const vpx_codec_err_t res =
        decoder_->DecodeFrame(video_->cxdata(), video_->frame_size());
    if (res == VPX_CODEC_OK) {
      ReadMd5();
      video_->Next();
    }
    return res;
  }

  vpx_codec_err_t DecodeRemainingFrames() {
    for (; video_->cxdata() != nullptr; video_->Next()) {
      const vpx_codec_err_t res =
          decoder_->DecodeFrame(video_->cxdata(), video_->frame_size());
      if (res != VPX_CODEC_OK) return res;
      ReadMd5();
    }
    return VPX_CODEC_OK;
  }

  // Checks if MD5 matches or doesn't.
  void CheckMd5(bool matches) {
    libvpx_test::DxDataIterator dec_iter = decoder_->GetDxData();
    const vpx_image_t *img = dec_iter.Next();
    CheckMd5Vpx(*img, matches);
  }

 private:
  // TODO(fgalligan): Move the MD5 testing code into another class.
  void OpenMd5File(const std::string &md5_file_name) {
    md5_file_ = libvpx_test::OpenTestDataFile(md5_file_name);
    ASSERT_NE(md5_file_, nullptr)
        << "MD5 file open failed. Filename: " << md5_file_name;
  }

  // Reads the next line of the MD5 file.
  void ReadMd5() {
    ASSERT_NE(md5_file_, nullptr);
    const int res = fscanf(md5_file_, "%s  %s", expected_md5_, junk_);
    ASSERT_NE(EOF, res) << "Read md5 data failed";
    expected_md5_[32] = '\0';
  }

  // Checks if the last read MD5 matches |img| or doesn't.
  void CheckMd5Vpx(const vpx_image_t &img, bool matches) {
    ::libvpx_test::MD5 md5_res;
    md5_res.Add(&img);
    const char *const actual_md5 = md5_res.Get();

    // Check MD5.
    if (matches)
      ASSERT_STREQ(expected_md5_, actual_md5) << "MD5 checksums don't match";
    else
      ASSERT_STRNE(expected_md5_, actual_md5) << "MD5 checksums match";
  }

  libvpx_test::WebMVideoSource *video_;
  libvpx_test::VP9Decoder *decoder_;
  FILE *md5_file_;
  char expected_md5_[33];
  char junk_[128];
};

TEST(SkipLoopFilterTest, ShutOffLoopFilter) {
  const int non_zero_value = 1;
  const int num_threads = 0;
  SkipLoopFilterTest skip_loop_filter;
  ASSERT_TRUE(skip_loop_filter.Init(num_threads));
  skip_loop_filter.SetSkipLoopFilter(non_zero_value, VPX_CODEC_OK);
  ASSERT_EQ(VPX_CODEC_OK, skip_loop_filter.DecodeRemainingFrames());
  skip_loop_filter.CheckMd5(false);
}

TEST(SkipLoopFilterTest, ShutOffLoopFilterSingleThread) {
  const int non_zero_value = 1;
  const int num_threads = 1;
  SkipLoopFilterTest skip_loop_filter;
  ASSERT_TRUE(skip_loop_filter.Init(num_threads));
  skip_loop_filter.SetSkipLoopFilter(non_zero_value, VPX_CODEC_OK);
  ASSERT_EQ(VPX_CODEC_OK, skip_loop_filter.DecodeRemainingFrames());
  skip_loop_filter.CheckMd5(false);
}

TEST(SkipLoopFilterTest, ShutOffLoopFilter8Threads) {
  const int non_zero_value = 1;
  const int num_threads = 8;
  SkipLoopFilterTest skip_loop_filter;
  ASSERT_TRUE(skip_loop_filter.Init(num_threads));
  skip_loop_filter.SetSkipLoopFilter(non_zero_value, VPX_CODEC_OK);
  ASSERT_EQ(VPX_CODEC_OK, skip_loop_filter.DecodeRemainingFrames());
  skip_loop_filter.CheckMd5(false);
}

TEST(SkipLoopFilterTest, WithLoopFilter) {
  const int non_zero_value = 1;
  const int num_threads = 0;
  SkipLoopFilterTest skip_loop_filter;
  ASSERT_TRUE(skip_loop_filter.Init(num_threads));
  skip_loop_filter.SetSkipLoopFilter(non_zero_value, VPX_CODEC_OK);
  skip_loop_filter.SetSkipLoopFilter(0, VPX_CODEC_OK);
  ASSERT_EQ(VPX_CODEC_OK, skip_loop_filter.DecodeRemainingFrames());
  skip_loop_filter.CheckMd5(true);
}

TEST(SkipLoopFilterTest, ToggleLoopFilter) {
  const int num_threads = 0;
  SkipLoopFilterTest skip_loop_filter;
  ASSERT_TRUE(skip_loop_filter.Init(num_threads));

  for (int i = 0; i < 10; ++i) {
    skip_loop_filter.SetSkipLoopFilter(i % 2, VPX_CODEC_OK);
    ASSERT_EQ(VPX_CODEC_OK, skip_loop_filter.DecodeOneFrame());
  }
  ASSERT_EQ(VPX_CODEC_OK, skip_loop_filter.DecodeRemainingFrames());
  skip_loop_filter.CheckMd5(false);
}

}  // namespace
