/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 1, 2022.
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
#include "aom_dsp/bitwriter.h"

void aom_start_encode(aom_writer *w, uint8_t *source) {
  w->buffer = source;
  w->pos = 0;
  od_ec_enc_init(&w->ec, 62025);
}

int aom_stop_encode(aom_writer *w) {
  int nb_bits;
  uint32_t bytes;
  unsigned char *data;
  data = od_ec_enc_done(&w->ec, &bytes);
  if (!data) {
    od_ec_enc_clear(&w->ec);
    return -1;
  }
  nb_bits = od_ec_enc_tell(&w->ec);
  memcpy(w->buffer, data, bytes);
  w->pos = bytes;
  od_ec_enc_clear(&w->ec);
  return nb_bits;
}

int aom_tell_size(aom_writer *w) {
  const int nb_bits = od_ec_enc_tell(&w->ec);
  return nb_bits;
}
