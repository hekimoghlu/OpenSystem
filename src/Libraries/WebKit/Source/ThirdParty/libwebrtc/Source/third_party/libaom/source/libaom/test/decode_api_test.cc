/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 24, 2025.
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
#include "gtest/gtest.h"

#include "config/aom_config.h"

#include "aom/aomdx.h"
#include "aom/aom_decoder.h"

namespace {

TEST(DecodeAPI, InvalidParams) {
  uint8_t buf[1] = { 0 };
  aom_codec_ctx_t dec;

  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_dec_init(nullptr, nullptr, nullptr, 0));
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_dec_init(&dec, nullptr, nullptr, 0));
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_decode(nullptr, nullptr, 0, nullptr));
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_decode(nullptr, buf, 0, nullptr));
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_decode(nullptr, buf, sizeof(buf), nullptr));
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_decode(nullptr, nullptr, sizeof(buf), nullptr));
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM, aom_codec_destroy(nullptr));
  EXPECT_NE(aom_codec_error(nullptr), nullptr);
  EXPECT_EQ(aom_codec_error_detail(nullptr), nullptr);

  aom_codec_iface_t *iface = aom_codec_av1_dx();
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_dec_init(nullptr, iface, nullptr, 0));

  EXPECT_EQ(AOM_CODEC_OK, aom_codec_dec_init(&dec, iface, nullptr, 0));
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM,
            aom_codec_decode(&dec, nullptr, sizeof(buf), nullptr));
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM, aom_codec_decode(&dec, buf, 0, nullptr));

  EXPECT_EQ(AOM_CODEC_OK, aom_codec_destroy(&dec));
}

TEST(DecodeAPI, InvalidControlId) {
  aom_codec_iface_t *iface = aom_codec_av1_dx();
  aom_codec_ctx_t dec;
  EXPECT_EQ(AOM_CODEC_OK, aom_codec_dec_init(&dec, iface, nullptr, 0));
  EXPECT_EQ(AOM_CODEC_ERROR, aom_codec_control(&dec, -1, 0));
  EXPECT_EQ(AOM_CODEC_INVALID_PARAM, aom_codec_control(&dec, 0, 0));
  EXPECT_EQ(AOM_CODEC_OK, aom_codec_destroy(&dec));
}

}  // namespace
