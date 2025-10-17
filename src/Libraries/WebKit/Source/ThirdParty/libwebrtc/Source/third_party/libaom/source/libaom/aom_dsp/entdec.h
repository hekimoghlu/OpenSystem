/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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
#ifndef AOM_AOM_DSP_ENTDEC_H_
#define AOM_AOM_DSP_ENTDEC_H_
#include <limits.h>
#include "aom_dsp/entcode.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct od_ec_dec od_ec_dec;

#if defined(OD_ACCOUNTING) && OD_ACCOUNTING
#define OD_ACC_STR , char *acc_str
#define od_ec_dec_bits(dec, ftb, str) od_ec_dec_bits_(dec, ftb, str)
#else
#define OD_ACC_STR
#define od_ec_dec_bits(dec, ftb, str) od_ec_dec_bits_(dec, ftb)
#endif

/*The entropy decoder context.*/
struct od_ec_dec {
  /*The start of the current input buffer.*/
  const unsigned char *buf;
  /*An offset used to keep track of tell after reaching the end of the stream.
    This is constant throughout most of the decoding process, but becomes
     important once we hit the end of the buffer and stop incrementing bptr
     (and instead pretend cnt has lots of bits).*/
  int32_t tell_offs;
  /*The end of the current input buffer.*/
  const unsigned char *end;
  /*The read pointer for the entropy-coded bits.*/
  const unsigned char *bptr;
  /*The difference between the high end of the current range, (low + rng), and
     the coded value, minus 1.
    This stores up to OD_EC_WINDOW_SIZE bits of that difference, but the
     decoder only uses the top 16 bits of the window to decode the next symbol.
    As we shift up during renormalization, if we don't have enough bits left in
     the window to fill the top 16, we'll read in more bits of the coded
     value.*/
  od_ec_window dif;
  /*The number of values in the current range.*/
  uint16_t rng;
  /*The number of bits of data in the current value.*/
  int16_t cnt;
};

/*See entdec.c for further documentation.*/

void od_ec_dec_init(od_ec_dec *dec, const unsigned char *buf, uint32_t storage)
    OD_ARG_NONNULL(1) OD_ARG_NONNULL(2);

OD_WARN_UNUSED_RESULT int od_ec_decode_bool_q15(od_ec_dec *dec, unsigned f)
    OD_ARG_NONNULL(1);
OD_WARN_UNUSED_RESULT int od_ec_decode_cdf_q15(od_ec_dec *dec,
                                               const uint16_t *cdf, int nsyms)
    OD_ARG_NONNULL(1) OD_ARG_NONNULL(2);

OD_WARN_UNUSED_RESULT uint32_t od_ec_dec_bits_(od_ec_dec *dec, unsigned ftb)
    OD_ARG_NONNULL(1);

OD_WARN_UNUSED_RESULT int od_ec_dec_tell(const od_ec_dec *dec)
    OD_ARG_NONNULL(1);
OD_WARN_UNUSED_RESULT uint32_t od_ec_dec_tell_frac(const od_ec_dec *dec)
    OD_ARG_NONNULL(1);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AOM_DSP_ENTDEC_H_
