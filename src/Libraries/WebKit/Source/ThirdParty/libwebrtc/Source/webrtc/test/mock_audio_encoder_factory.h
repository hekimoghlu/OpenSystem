/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 18, 2022.
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
#ifndef TEST_MOCK_AUDIO_ENCODER_FACTORY_H_
#define TEST_MOCK_AUDIO_ENCODER_FACTORY_H_

#include <memory>
#include <vector>

#include "api/audio_codecs/audio_encoder_factory.h"
#include "api/environment/environment.h"
#include "api/make_ref_counted.h"
#include "api/scoped_refptr.h"
#include "test/gmock.h"

namespace webrtc {

class MockAudioEncoderFactory
    : public ::testing::NiceMock<AudioEncoderFactory> {
 public:
  MOCK_METHOD(std::vector<AudioCodecSpec>,
              GetSupportedEncoders,
              (),
              (override));
  MOCK_METHOD(std::optional<AudioCodecInfo>,
              QueryAudioEncoder,
              (const SdpAudioFormat& format),
              (override));
  MOCK_METHOD(std::unique_ptr<AudioEncoder>,
              Create,
              (const Environment&, const SdpAudioFormat&, Options),
              (override));

  // Creates a MockAudioEncoderFactory with no formats and that may not be
  // invoked to create a codec - useful for initializing a voice engine, for
  // example.
  static scoped_refptr<MockAudioEncoderFactory> CreateUnusedFactory() {
    auto factory = make_ref_counted<MockAudioEncoderFactory>();
    EXPECT_CALL(*factory, Create).Times(0);
    return factory;
  }

  // Creates a MockAudioEncoderFactory with no formats that may be invoked to
  // create a codec any number of times. It will, though, return nullptr on each
  // call, since it supports no codecs.
  static scoped_refptr<MockAudioEncoderFactory> CreateEmptyFactory() {
    return make_ref_counted<MockAudioEncoderFactory>();
  }
};

}  // namespace webrtc

#endif  // TEST_MOCK_AUDIO_ENCODER_FACTORY_H_
