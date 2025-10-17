/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 24, 2023.
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
#ifndef VPX_TEST_MD5_HELPER_H_
#define VPX_TEST_MD5_HELPER_H_

#include "./md5_utils.h"
#include "vpx/vpx_decoder.h"

namespace libvpx_test {
class MD5 {
 public:
  MD5() { MD5Init(&md5_); }

  void Add(const vpx_image_t *img) {
    for (int plane = 0; plane < 3; ++plane) {
      const uint8_t *buf = img->planes[plane];
      // Calculate the width and height to do the md5 check. For the chroma
      // plane, we never want to round down and thus skip a pixel so if
      // we are shifting by 1 (chroma_shift) we add 1 before doing the shift.
      // This works only for chroma_shift of 0 and 1.
      const int bytes_per_sample =
          (img->fmt & VPX_IMG_FMT_HIGHBITDEPTH) ? 2 : 1;
      const int h =
          plane ? (img->d_h + img->y_chroma_shift) >> img->y_chroma_shift
                : img->d_h;
      const int w =
          (plane ? (img->d_w + img->x_chroma_shift) >> img->x_chroma_shift
                 : img->d_w) *
          bytes_per_sample;

      for (int y = 0; y < h; ++y) {
        MD5Update(&md5_, buf, w);
        buf += img->stride[plane];
      }
    }
  }

  void Add(const uint8_t *data, size_t size) {
    MD5Update(&md5_, data, static_cast<uint32_t>(size));
  }

  const char *Get() {
    static const char hex[16] = {
      '0', '1', '2', '3', '4', '5', '6', '7',
      '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
    };
    uint8_t tmp[16];
    MD5Context ctx_tmp = md5_;

    MD5Final(tmp, &ctx_tmp);
    for (int i = 0; i < 16; i++) {
      res_[i * 2 + 0] = hex[tmp[i] >> 4];
      res_[i * 2 + 1] = hex[tmp[i] & 0xf];
    }
    res_[32] = 0;

    return res_;
  }

 protected:
  char res_[33];
  MD5Context md5_;
};

}  // namespace libvpx_test

#endif  // VPX_TEST_MD5_HELPER_H_
