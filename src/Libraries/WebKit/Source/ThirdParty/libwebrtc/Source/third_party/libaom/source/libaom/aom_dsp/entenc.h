/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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
#ifndef AOM_AOM_DSP_ENTENC_H_
#define AOM_AOM_DSP_ENTENC_H_
#include <stddef.h>
#include "aom_dsp/entcode.h"
#include "aom_util/endian_inl.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t od_ec_enc_window;

typedef struct od_ec_enc od_ec_enc;

#define OD_MEASURE_EC_OVERHEAD (0)

/*The entropy encoder context.*/
struct od_ec_enc {
  /*Buffered output.
    This contains only the raw bits until the final call to od_ec_enc_done(),
     where all the arithmetic-coded data gets prepended to it.*/
  unsigned char *buf;
  /*The size of the buffer.*/
  uint32_t storage;
  /*The offset at which the next entropy-coded byte will be written.*/
  uint32_t offs;
  /*The low end of the current range.*/
  od_ec_enc_window low;
  /*The number of values in the current range.*/
  uint16_t rng;
  /*The number of bits of data in the current value.*/
  int16_t cnt;
  /*Nonzero if an error occurred.*/
  int error;
#if OD_MEASURE_EC_OVERHEAD
  double entropy;
  int nb_symbols;
#endif
};

/*See entenc.c for further documentation.*/

void od_ec_enc_init(od_ec_enc *enc, uint32_t size) OD_ARG_NONNULL(1);
void od_ec_enc_reset(od_ec_enc *enc) OD_ARG_NONNULL(1);
void od_ec_enc_clear(od_ec_enc *enc) OD_ARG_NONNULL(1);

void od_ec_encode_bool_q15(od_ec_enc *enc, int val, unsigned f_q15)
    OD_ARG_NONNULL(1);
void od_ec_encode_cdf_q15(od_ec_enc *enc, int s, const uint16_t *cdf, int nsyms)
    OD_ARG_NONNULL(1) OD_ARG_NONNULL(3);

void od_ec_enc_bits(od_ec_enc *enc, uint32_t fl, unsigned ftb)
    OD_ARG_NONNULL(1);

OD_WARN_UNUSED_RESULT unsigned char *od_ec_enc_done(od_ec_enc *enc,
                                                    uint32_t *nbytes)
    OD_ARG_NONNULL(1) OD_ARG_NONNULL(2);

OD_WARN_UNUSED_RESULT int od_ec_enc_tell(const od_ec_enc *enc)
    OD_ARG_NONNULL(1);
OD_WARN_UNUSED_RESULT uint32_t od_ec_enc_tell_frac(const od_ec_enc *enc)
    OD_ARG_NONNULL(1);

// buf is the frame bitbuffer, offs is where carry to be added
static inline void propagate_carry_bwd(unsigned char *buf, uint32_t offs) {
  uint16_t sum, carry = 1;
  do {
    sum = (uint16_t)buf[offs] + 1;
    buf[offs--] = (unsigned char)sum;
    carry = sum >> 8;
  } while (carry);
}

// Convert to big-endian byte order and write data to buffer adding the
// carry-bit
static inline void write_enc_data_to_out_buf(unsigned char *out, uint32_t offs,
                                             uint64_t output, uint64_t carry,
                                             uint32_t *enc_offs,
                                             uint8_t num_bytes_ready) {
  const uint64_t reg = HToBE64(output << ((8 - num_bytes_ready) << 3));
  memcpy(&out[offs], &reg, 8);
  // Propagate carry backwards if exists
  if (carry) {
    assert(offs > 0);
    propagate_carry_bwd(out, offs - 1);
  }
  *enc_offs = offs + num_bytes_ready;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AOM_DSP_ENTENC_H_
