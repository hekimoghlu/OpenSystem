/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 20, 2021.
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
#include "modules/video_coding/decoder_database.h"

#include <memory>
#include <utility>

#include "api/test/mock_video_decoder.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

using ::testing::NiceMock;

// Test registering and unregistering an external decoder instance.
TEST(VCMDecoderDatabaseTest, RegisterExternalDecoder) {
  VCMDecoderDatabase db;
  constexpr int kPayloadType = 1;
  ASSERT_FALSE(db.IsExternalDecoderRegistered(kPayloadType));

  auto decoder = std::make_unique<NiceMock<MockVideoDecoder>>();
  bool decoder_deleted = false;
  EXPECT_CALL(*decoder, Destruct).WillOnce([&decoder_deleted] {
    decoder_deleted = true;
  });

  db.RegisterExternalDecoder(kPayloadType, std::move(decoder));
  EXPECT_TRUE(db.IsExternalDecoderRegistered(kPayloadType));
  db.DeregisterExternalDecoder(kPayloadType);
  EXPECT_TRUE(decoder_deleted);
  EXPECT_FALSE(db.IsExternalDecoderRegistered(kPayloadType));
}

TEST(VCMDecoderDatabaseTest, RegisterReceiveCodec) {
  VCMDecoderDatabase db;
  constexpr int kPayloadType = 1;
  ASSERT_FALSE(db.DeregisterReceiveCodec(kPayloadType));

  VideoDecoder::Settings settings;
  settings.set_codec_type(kVideoCodecVP8);
  settings.set_max_render_resolution({10, 10});
  settings.set_number_of_cores(4);
  db.RegisterReceiveCodec(kPayloadType, settings);

  EXPECT_TRUE(db.DeregisterReceiveCodec(kPayloadType));
}

TEST(VCMDecoderDatabaseTest, DeregisterReceiveCodecs) {
  VCMDecoderDatabase db;
  constexpr int kPayloadType1 = 1;
  constexpr int kPayloadType2 = 2;
  ASSERT_FALSE(db.DeregisterReceiveCodec(kPayloadType1));
  ASSERT_FALSE(db.DeregisterReceiveCodec(kPayloadType2));

  VideoDecoder::Settings settings1;
  settings1.set_codec_type(kVideoCodecVP8);
  settings1.set_max_render_resolution({10, 10});
  settings1.set_number_of_cores(4);

  VideoDecoder::Settings settings2 = settings1;
  settings2.set_codec_type(kVideoCodecVP9);

  db.RegisterReceiveCodec(kPayloadType1, settings1);
  db.RegisterReceiveCodec(kPayloadType2, settings2);

  db.DeregisterReceiveCodecs();

  // All receive codecs must have been removed.
  EXPECT_FALSE(db.DeregisterReceiveCodec(kPayloadType1));
  EXPECT_FALSE(db.DeregisterReceiveCodec(kPayloadType2));
}

}  // namespace
}  // namespace webrtc
