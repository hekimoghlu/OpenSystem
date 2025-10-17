/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_MOCK_MOCK_DECODER_DATABASE_H_
#define MODULES_AUDIO_CODING_NETEQ_MOCK_MOCK_DECODER_DATABASE_H_

#include <string>

#include "api/environment/environment_factory.h"
#include "modules/audio_coding/neteq/decoder_database.h"
#include "test/gmock.h"

namespace webrtc {

class MockDecoderDatabase : public DecoderDatabase {
 public:
  MockDecoderDatabase()
      : DecoderDatabase(CreateEnvironment(),
                        /*decoder_factory=*/nullptr,
                        /*codec_pair_id=*/std::nullopt) {}
  ~MockDecoderDatabase() override { Die(); }
  MOCK_METHOD(void, Die, ());
  MOCK_METHOD(bool, Empty, (), (const, override));
  MOCK_METHOD(int, Size, (), (const, override));
  MOCK_METHOD(int,
              RegisterPayload,
              (int rtp_payload_type, const SdpAudioFormat& audio_format),
              (override));
  MOCK_METHOD(int, Remove, (uint8_t rtp_payload_type), (override));
  MOCK_METHOD(void, RemoveAll, (), (override));
  MOCK_METHOD(const DecoderInfo*,
              GetDecoderInfo,
              (uint8_t rtp_payload_type),
              (const, override));
  MOCK_METHOD(int,
              SetActiveDecoder,
              (uint8_t rtp_payload_type, bool* new_decoder),
              (override));
  MOCK_METHOD(AudioDecoder*, GetActiveDecoder, (), (const, override));
  MOCK_METHOD(int, SetActiveCngDecoder, (uint8_t rtp_payload_type), (override));
  MOCK_METHOD(ComfortNoiseDecoder*, GetActiveCngDecoder, (), (const, override));
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_MOCK_MOCK_DECODER_DATABASE_H_
