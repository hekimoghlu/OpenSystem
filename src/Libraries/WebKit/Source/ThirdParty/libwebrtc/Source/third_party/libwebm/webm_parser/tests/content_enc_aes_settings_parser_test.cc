/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 26, 2022.
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

// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "src/content_enc_aes_settings_parser.h"

#include "gtest/gtest.h"

#include "test_utils/element_parser_test.h"
#include "webm/id.h"

using webm::AesSettingsCipherMode;
using webm::ContentEncAesSettings;
using webm::ContentEncAesSettingsParser;
using webm::ElementParserTest;
using webm::Id;

namespace {

class ContentEncAesSettingsParserTest
    : public ElementParserTest<ContentEncAesSettingsParser,
                               Id::kContentEncAesSettings> {};

TEST_F(ContentEncAesSettingsParserTest, DefaultParse) {
  ParseAndVerify();

  const ContentEncAesSettings content_enc_aes_settings = parser_.value();

  EXPECT_FALSE(content_enc_aes_settings.aes_settings_cipher_mode.is_present());
  EXPECT_EQ(AesSettingsCipherMode::kCtr,
            content_enc_aes_settings.aes_settings_cipher_mode.value());
}

TEST_F(ContentEncAesSettingsParserTest, DefaultValues) {
  SetReaderData({
      0x47, 0xE8,  // ID = 0x47E8 (AESSettingsCipherMode).
      0x80,  // Size = 0.
  });

  ParseAndVerify();

  const ContentEncAesSettings content_enc_aes_settings = parser_.value();

  EXPECT_TRUE(content_enc_aes_settings.aes_settings_cipher_mode.is_present());
  EXPECT_EQ(AesSettingsCipherMode::kCtr,
            content_enc_aes_settings.aes_settings_cipher_mode.value());
}

TEST_F(ContentEncAesSettingsParserTest, CustomValues) {
  SetReaderData({
      0x47, 0xE8,  // ID = 0x47E8 (AESSettingsCipherMode).
      0x81,  // Size = 1.
      0x00,  // Body (value = 0).
  });

  ParseAndVerify();

  const ContentEncAesSettings content_enc_aes_settings = parser_.value();

  EXPECT_TRUE(content_enc_aes_settings.aes_settings_cipher_mode.is_present());
  EXPECT_EQ(static_cast<AesSettingsCipherMode>(0),
            content_enc_aes_settings.aes_settings_cipher_mode.value());
}

}  // namespace
