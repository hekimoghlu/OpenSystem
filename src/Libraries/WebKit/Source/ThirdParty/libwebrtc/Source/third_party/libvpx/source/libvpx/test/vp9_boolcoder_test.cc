/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "gtest/gtest.h"

#include "test/acm_random.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/bitreader.h"
#include "vpx_dsp/bitwriter.h"

using libvpx_test::ACMRandom;

namespace {
const int num_tests = 10;
}  // namespace

TEST(VP9, TestBitIO) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  for (int n = 0; n < num_tests; ++n) {
    for (int method = 0; method <= 7; ++method) {  // we generate various proba
      const int kBitsToTest = 1000;
      uint8_t probas[kBitsToTest];

      for (int i = 0; i < kBitsToTest; ++i) {
        const int parity = i & 1;
        /* clang-format off */
        probas[i] =
          (method == 0) ? 0 : (method == 1) ? 255 :
          (method == 2) ? 128 :
          (method == 3) ? rnd.Rand8() :
          (method == 4) ? (parity ? 0 : 255) :
            // alternate between low and high proba:
            (method == 5) ? (parity ? rnd(128) : 255 - rnd(128)) :
            (method == 6) ?
            (parity ? rnd(64) : 255 - rnd(64)) :
            (parity ? rnd(32) : 255 - rnd(32));
        /* clang-format on */
      }
      for (int bit_method = 0; bit_method <= 3; ++bit_method) {
        const int random_seed = 6432;
        const int kBufferSize = 10000;
        ACMRandom bit_rnd(random_seed);
        vpx_writer bw;
        uint8_t bw_buffer[kBufferSize];
        vpx_start_encode(&bw, bw_buffer, sizeof(bw_buffer));

        int bit = (bit_method == 0) ? 0 : (bit_method == 1) ? 1 : 0;
        for (int i = 0; i < kBitsToTest; ++i) {
          if (bit_method == 2) {
            bit = (i & 1);
          } else if (bit_method == 3) {
            bit = bit_rnd(2);
          }
          vpx_write(&bw, bit, static_cast<int>(probas[i]));
        }

        GTEST_ASSERT_EQ(vpx_stop_encode(&bw), 0);
        // vpx_reader_fill() may read into uninitialized data that
        // isn't used meaningfully, but may trigger an MSan warning.
        memset(bw_buffer + bw.pos, 0, sizeof(BD_VALUE) - 1);

        // First bit should be zero
        GTEST_ASSERT_EQ(bw_buffer[0] & 0x80, 0);

        vpx_reader br;
        vpx_reader_init(&br, bw_buffer, kBufferSize, nullptr, nullptr);
        bit_rnd.Reset(random_seed);
        for (int i = 0; i < kBitsToTest; ++i) {
          if (bit_method == 2) {
            bit = (i & 1);
          } else if (bit_method == 3) {
            bit = bit_rnd(2);
          }
          GTEST_ASSERT_EQ(vpx_read(&br, probas[i]), bit)
              << "pos: " << i << " / " << kBitsToTest
              << " bit_method: " << bit_method << " method: " << method;
        }
      }
    }
  }
}

TEST(VP9, TestBitIOBufferSize0) {
  vpx_writer bw;
  uint8_t bw_buffer[1];
  vpx_start_encode(&bw, bw_buffer, 0);
  GTEST_ASSERT_EQ(vpx_stop_encode(&bw), -1);
}

TEST(VP9, TestBitIOBufferSize1) {
  vpx_writer bw;
  uint8_t bw_buffer[1];
  vpx_start_encode(&bw, bw_buffer, sizeof(bw_buffer));
  GTEST_ASSERT_EQ(vpx_stop_encode(&bw), -1);
}

TEST(VP9, TestBitIOBufferSize2) {
  vpx_writer bw;
  uint8_t bw_buffer[2];
  vpx_start_encode(&bw, bw_buffer, sizeof(bw_buffer));
  GTEST_ASSERT_EQ(vpx_stop_encode(&bw), 0);
}
