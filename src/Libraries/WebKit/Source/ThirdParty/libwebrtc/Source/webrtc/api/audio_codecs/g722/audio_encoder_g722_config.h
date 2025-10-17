/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 5, 2024.
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
#ifndef API_AUDIO_CODECS_G722_AUDIO_ENCODER_G722_CONFIG_H_
#define API_AUDIO_CODECS_G722_AUDIO_ENCODER_G722_CONFIG_H_

#include "api/audio_codecs/audio_encoder.h"

namespace webrtc {

struct AudioEncoderG722Config {
  bool IsOk() const {
    return frame_size_ms > 0 && frame_size_ms % 10 == 0 && num_channels >= 1 &&
           num_channels <= AudioEncoder::kMaxNumberOfChannels;
  }
  int frame_size_ms = 20;
  int num_channels = 1;
};

}  // namespace webrtc

#endif  // API_AUDIO_CODECS_G722_AUDIO_ENCODER_G722_CONFIG_H_
