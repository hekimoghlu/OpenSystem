/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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

//  Created by rafirafi on 3/17/16.
//  Copyright (c) 2016 rafirafi. All rights reserved.

#ifndef PDCRYPTO_DUMMY_H
#define PDCRYPTO_DUMMY_H

#include <corecrypto/ccdigest.h>
#include <corecrypto/cchmac.h>
#include <corecrypto/ccmode.h>
#include <corecrypto/ccrc4.h>
#include <corecrypto/ccrng.h>
#include <corecrypto/ccrsa.h>

void pdcdigest_final_fn_dummy(const struct ccdigest_info *di,
                              ccdigest_ctx_t ctx,
                              void *digest);

void pdcdigest_fn_dummy(const struct ccdigest_info *di,
                        unsigned long len,
                        const void *data, void *digest);

extern const struct ccdigest_info pdcsha384_di_dummy;
extern const struct ccdigest_info pdcsha512_di_dummy;

void pdchmac_init_fn_dummy(const struct ccdigest_info *di,
                                 cchmac_ctx_t ctx,
                                 unsigned long key_len, const void *key);

void pdchmac_update_fn_dummy(const struct ccdigest_info *di,
                                   cchmac_ctx_t ctx,
                                   unsigned long data_len,
                                   const void *data);

void pdchmac_final_fn_dummy(const struct ccdigest_info *di,
                                  cchmac_ctx_t ctx,
                                  unsigned char *mac);

void pdchmac_fn_dummy(const struct ccdigest_info *di,
                            unsigned long key_len,
                            const void *key,
                            unsigned long data_len,
                            const void *data,
                            unsigned char *mac);

extern const struct ccmode_ecb pdcaes_ecb_encrypt_dummy;
extern const struct ccmode_ecb pdcaes_ecb_decrypt_dummy;
extern const struct ccmode_cbc pdcaes_cbc_encrypt_dummy;
extern const struct ccmode_cbc pdcaes_cbc_decrypt_dummy;
extern const struct ccmode_xts pdcaes_xts_encrypt_dummy;
extern const struct ccmode_xts pdcaes_xts_decrypt_dummy;
extern const struct ccmode_gcm pdcaes_gcm_encrypt_dummy;
extern const struct ccmode_gcm pdcaes_gcm_decrypt_dummy;

extern const struct ccmode_ecb pdcdes_ecb_encrypt_dummy;
extern const struct ccmode_ecb pdcdes_ecb_decrypt_dummy;
extern const struct ccmode_cbc pdcdes_cbc_encrypt_dummy;
extern const struct ccmode_cbc pdcdes_cbc_decrypt_dummy;

extern const struct ccmode_ecb pdctdes_ecb_encrypt_dummy;
extern const struct ccmode_ecb pdctdes_ecb_decrypt_dummy;
extern const struct ccmode_cbc pdctdes_cbc_encrypt_dummy;
extern const struct ccmode_cbc pdctdes_cbc_decrypt_dummy;

extern const struct ccrc4_info pdcrc4_info_dummy;

extern const struct ccmode_ecb pdcblowfish_ecb_encrypt_dummy;
extern const struct ccmode_ecb pdcblowfish_ecb_decrypt_dummy;

extern const struct ccmode_ecb pdccast_ecb_encrypt_dummy;
extern const struct ccmode_ecb pdccast_ecb_decrypt_dummy;

int pdcdes_key_is_weak_fn_dummy(void *key,
                                 unsigned long  length);

void pdcdes_key_set_odd_parity_fn_dummy(void *key,
                                         unsigned long length);


void pdcpad_xts_decrypt_fn_dummy(const struct ccmode_xts *xts,
                                  ccxts_ctx *ctx,
                                  unsigned long nbytes,
                                  const void *in,
                                  void *out);

void pdcpad_xts_encrypt_fn_dummy(const struct ccmode_xts *xts,
                                  ccxts_ctx *ctx,
                                  unsigned long nbytes,
                                  const void *in,
                                  void *out);
#endif // PDCRYPTO_DUMMY_H
