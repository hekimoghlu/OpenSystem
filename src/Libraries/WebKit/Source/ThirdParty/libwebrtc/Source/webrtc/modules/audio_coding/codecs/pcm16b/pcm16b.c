/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 4, 2023.
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
#include "modules/audio_coding/codecs/pcm16b/pcm16b.h"

size_t WebRtcPcm16b_Encode(const int16_t* speech,
                           size_t len,
                           uint8_t* encoded) {
  size_t i;
  for (i = 0; i < len; ++i) {
    uint16_t s = speech[i];
    encoded[2 * i] = s >> 8;
    encoded[2 * i + 1] = s;
  }
  return 2 * len;
}

size_t WebRtcPcm16b_Decode(const uint8_t* encoded,
                           size_t len,
                           int16_t* speech) {
  size_t i;
  for (i = 0; i < len / 2; ++i)
    speech[i] = encoded[2 * i] << 8 | encoded[2 * i + 1];
  return len / 2;
}
