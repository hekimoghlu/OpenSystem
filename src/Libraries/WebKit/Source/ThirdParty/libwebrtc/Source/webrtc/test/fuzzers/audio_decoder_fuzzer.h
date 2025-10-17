/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 15, 2022.
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
#ifndef TEST_FUZZERS_AUDIO_DECODER_FUZZER_H_
#define TEST_FUZZERS_AUDIO_DECODER_FUZZER_H_

#include <stddef.h>
#include <stdint.h>

namespace webrtc {

class AudioDecoder;

enum class DecoderFunctionType {
  kNormalDecode,
  kRedundantDecode,
};

void FuzzAudioDecoder(DecoderFunctionType decode_type,
                      const uint8_t* data,
                      size_t size,
                      AudioDecoder* decoder,
                      int sample_rate_hz,
                      size_t max_decoded_bytes,
                      int16_t* decoded);

}  // namespace webrtc

#endif  // TEST_FUZZERS_AUDIO_DECODER_FUZZER_H_
