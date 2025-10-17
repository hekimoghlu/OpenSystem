/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 26, 2024.
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
#include <string.h>

#include "modules/third_party/g711/g711.h"
#include "modules/audio_coding/codecs/g711/g711_interface.h"

size_t WebRtcG711_EncodeA(const int16_t* speechIn,
                          size_t len,
                          uint8_t* encoded) {
  size_t n;
  for (n = 0; n < len; n++)
    encoded[n] = linear_to_alaw(speechIn[n]);
  return len;
}

size_t WebRtcG711_EncodeU(const int16_t* speechIn,
                          size_t len,
                          uint8_t* encoded) {
  size_t n;
  for (n = 0; n < len; n++)
    encoded[n] = linear_to_ulaw(speechIn[n]);
  return len;
}

size_t WebRtcG711_DecodeA(const uint8_t* encoded,
                          size_t len,
                          int16_t* decoded,
                          int16_t* speechType) {
  size_t n;
  for (n = 0; n < len; n++)
    decoded[n] = alaw_to_linear(encoded[n]);
  *speechType = 1;
  return len;
}

size_t WebRtcG711_DecodeU(const uint8_t* encoded,
                          size_t len,
                          int16_t* decoded,
                          int16_t* speechType) {
  size_t n;
  for (n = 0; n < len; n++)
    decoded[n] = ulaw_to_linear(encoded[n]);
  *speechType = 1;
  return len;
}

int16_t WebRtcG711_Version(char* version, int16_t lenBytes) {
  strncpy(version, "2.0.0", lenBytes);
  return 0;
}
