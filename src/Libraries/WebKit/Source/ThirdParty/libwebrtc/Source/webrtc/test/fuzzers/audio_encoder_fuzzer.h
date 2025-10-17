/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 15, 2025.
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
#ifndef TEST_FUZZERS_AUDIO_ENCODER_FUZZER_H_
#define TEST_FUZZERS_AUDIO_ENCODER_FUZZER_H_

#include <memory>

#include "api/array_view.h"
#include "api/audio_codecs/audio_encoder.h"

namespace webrtc {

void FuzzAudioEncoder(rtc::ArrayView<const uint8_t> data_view,
                      std::unique_ptr<AudioEncoder> encoder);

}  // namespace webrtc

#endif  // TEST_FUZZERS_AUDIO_ENCODER_FUZZER_H_
