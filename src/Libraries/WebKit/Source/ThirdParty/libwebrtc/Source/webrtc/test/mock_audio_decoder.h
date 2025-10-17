/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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
#ifndef TEST_MOCK_AUDIO_DECODER_H_
#define TEST_MOCK_AUDIO_DECODER_H_

#include "api/audio_codecs/audio_decoder.h"
#include "test/gmock.h"

namespace webrtc {

class MockAudioDecoder : public AudioDecoder {
 public:
  MockAudioDecoder();
  ~MockAudioDecoder();
  MOCK_METHOD(void, Die, ());
  MOCK_METHOD(int,
              DecodeInternal,
              (const uint8_t*, size_t, int, int16_t*, SpeechType*),
              (override));
  MOCK_METHOD(bool, HasDecodePlc, (), (const, override));
  MOCK_METHOD(size_t, DecodePlc, (size_t, int16_t*), (override));
  MOCK_METHOD(void, Reset, (), (override));
  MOCK_METHOD(int, ErrorCode, (), (override));
  MOCK_METHOD(int, PacketDuration, (const uint8_t*, size_t), (const, override));
  MOCK_METHOD(size_t, Channels, (), (const, override));
  MOCK_METHOD(int, SampleRateHz, (), (const, override));
};

}  // namespace webrtc
#endif  // TEST_MOCK_AUDIO_DECODER_H_
