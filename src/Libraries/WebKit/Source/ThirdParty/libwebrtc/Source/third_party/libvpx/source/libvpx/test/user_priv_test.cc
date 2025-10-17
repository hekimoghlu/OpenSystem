/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 10, 2024.
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
#include <cstdio>
#include <cstdlib>
#include <string>
#include "gtest/gtest.h"
#include "./vpx_config.h"
#include "test/acm_random.h"
#include "test/codec_factory.h"
#include "test/decode_test_driver.h"
#include "test/ivf_video_source.h"
#include "test/md5_helper.h"
#include "test/util.h"
#if CONFIG_WEBM_IO
#include "test/webm_video_source.h"
#endif
#include "vpx_mem/vpx_mem.h"
#include "vpx/vp8.h"

namespace {

using libvpx_test::ACMRandom;
using std::string;

#if CONFIG_WEBM_IO

void CheckUserPrivateData(void *user_priv, int *target) {
  // actual pointer value should be the same as expected.
  EXPECT_EQ(reinterpret_cast<void *>(target), user_priv)
      << "user_priv pointer value does not match.";
}

// Decodes |filename|. Passes in user_priv data when calling DecodeFrame and
// compares the user_priv from return img with the original user_priv to see if
// they match. Both the pointer values and the values inside the addresses
// should match.
string DecodeFile(const string &filename) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  libvpx_test::WebMVideoSource video(filename);
  video.Init();

  vpx_codec_dec_cfg_t cfg = vpx_codec_dec_cfg_t();
  libvpx_test::VP9Decoder decoder(cfg, 0);

  libvpx_test::MD5 md5;
  int frame_num = 0;
  for (video.Begin(); !::testing::Test::HasFailure() && video.cxdata();
       video.Next()) {
    void *user_priv = reinterpret_cast<void *>(&frame_num);
    const vpx_codec_err_t res =
        decoder.DecodeFrame(video.cxdata(), video.frame_size(),
                            (frame_num == 0) ? nullptr : user_priv);
    if (res != VPX_CODEC_OK) {
      EXPECT_EQ(VPX_CODEC_OK, res) << decoder.DecodeError();
      break;
    }
    libvpx_test::DxDataIterator dec_iter = decoder.GetDxData();
    const vpx_image_t *img = nullptr;

    // Get decompressed data.
    while ((img = dec_iter.Next())) {
      if (frame_num == 0) {
        CheckUserPrivateData(img->user_priv, nullptr);
      } else {
        CheckUserPrivateData(img->user_priv, &frame_num);

        // Also test ctrl_get_reference api.
        struct vp9_ref_frame ref = vp9_ref_frame();
        // Randomly fetch a reference frame.
        ref.idx = rnd.Rand8() % 3;
        decoder.Control(VP9_GET_REFERENCE, &ref);

        CheckUserPrivateData(ref.img.user_priv, nullptr);
      }
      md5.Add(img);
    }

    frame_num++;
  }
  return string(md5.Get());
}

TEST(UserPrivTest, VideoDecode) {
  // no tiles or frame parallel; this exercises the decoding to test the
  // user_priv.
  EXPECT_STREQ("b35a1b707b28e82be025d960aba039bc",
               DecodeFile("vp90-2-03-size-226x226.webm").c_str());
}

#endif  // CONFIG_WEBM_IO

}  // namespace
