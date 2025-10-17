/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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
#include <vector>
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/ivf_video_source.h"

namespace {
// In a real use the 'decrypt_state' parameter will be a pointer to a struct
// with whatever internal state the decryptor uses. For testing we'll just
// xor with a constant key, and decrypt_state will point to the start of
// the original buffer.
const uint8_t test_key[16] = { 0x01, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
                               0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0xf0 };

void encrypt_buffer(const uint8_t *src, uint8_t *dst, size_t size,
                    ptrdiff_t offset) {
  for (size_t i = 0; i < size; ++i) {
    dst[i] = src[i] ^ test_key[(offset + i) & 15];
  }
}

void test_decrypt_cb(void *decrypt_state, const uint8_t *input, uint8_t *output,
                     int count) {
  encrypt_buffer(input, output, count,
                 input - reinterpret_cast<uint8_t *>(decrypt_state));
}

}  // namespace

namespace libvpx_test {

TEST(TestDecrypt, DecryptWorksVp8) {
  libvpx_test::IVFVideoSource video("vp80-00-comprehensive-001.ivf");
  video.Init();

  vpx_codec_dec_cfg_t dec_cfg = vpx_codec_dec_cfg_t();
  VP8Decoder decoder(dec_cfg, 0);

  video.Begin();

  // no decryption
  vpx_codec_err_t res = decoder.DecodeFrame(video.cxdata(), video.frame_size());
  ASSERT_EQ(VPX_CODEC_OK, res) << decoder.DecodeError();

  // decrypt frame
  video.Next();

  std::vector<uint8_t> encrypted(video.frame_size());
  encrypt_buffer(video.cxdata(), &encrypted[0], video.frame_size(), 0);
  vpx_decrypt_init di = { test_decrypt_cb, &encrypted[0] };
  decoder.Control(VPXD_SET_DECRYPTOR, &di);

  res = decoder.DecodeFrame(&encrypted[0], encrypted.size());
  ASSERT_EQ(VPX_CODEC_OK, res) << decoder.DecodeError();
}

}  // namespace libvpx_test
