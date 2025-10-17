/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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
#include <sys/appleapiopts.h>

#ifdef BSD_KERNEL_PRIVATE

#ifndef _ESP_CHACHA_POLY_H_
#define _ESP_CHACHA_POLY_H_

#define ESP_CHACHAPOLY_PAD_BOUND                        1
#define ESP_CHACHAPOLY_IV_LEN                           8
#define ESP_CHACHAPOLY_ICV_LEN                          16
#define ESP_CHACHAPOLY_KEYBITS_WITH_SALT        288 /* 32 bytes key + 4 bytes salt */

size_t esp_chachapoly_schedlen(const struct esp_algorithm *);
int esp_chachapoly_schedule(const struct esp_algorithm *,
    struct secasvar *);
int esp_chachapoly_encrypt(struct mbuf *, size_t, size_t, struct secasvar *,
    const struct esp_algorithm *, int);
int esp_chachapoly_decrypt(struct mbuf *, size_t, struct secasvar *,
    const struct esp_algorithm *, int);
int esp_chachapoly_encrypt_data(struct secasvar *,
    uint8_t *__sized_by(input_data_len), size_t input_data_len,
    struct newesp *,
    uint8_t *__sized_by(out_ivlen), size_t out_ivlen,
    uint8_t *__sized_by(output_data_len), size_t output_data_len);
int esp_chachapoly_decrypt_data(struct secasvar *,
    uint8_t *__sized_by(input_data_len), size_t input_data_len,
    struct newesp *,
    uint8_t *__sized_by(ivlen), size_t ivlen,
    uint8_t *__sized_by(output_data_len), size_t output_data_len);
int esp_chachapoly_encrypt_finalize(struct secasvar *, unsigned char *, size_t);
int esp_chachapoly_decrypt_finalize(struct secasvar *, unsigned char *, size_t);
int esp_chachapoly_mature(struct secasvar *);
int esp_chachapoly_ivlen(const struct esp_algorithm *, struct secasvar *);

#endif /* _ESP_CHACHA_POLY_H_ */
#endif /* BSD_KERNEL_PRIVATE */
