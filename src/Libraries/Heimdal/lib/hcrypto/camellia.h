/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 5, 2025.
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
/* $Id$ */

#ifndef HEIM_CAMELLIA_H
#define HEIM_CAMELLIA_H 1

/* symbol renaming */
#define CAMELLIA_set_key hc_CAMELLIA_set_encrypt_key
#define CAMELLIA_encrypt hc_CAMELLIA_encrypt
#define CAMELLIA_decrypt hc_CAMELLIA_decrypt
#define CAMELLIA_cbc_encrypt hc_CAMELLIA_cbc_encrypt

/*
 *
 */

#define CAMELLIA_BLOCK_SIZE 16
#define CAMELLIA_TABLE_BYTE_LEN 272
#define CAMELLIA_TABLE_WORD_LEN (CAMELLIA_TABLE_BYTE_LEN / 4)

#define CAMELLIA_ENCRYPT 1
#define CAMELLIA_DECRYPT 0

typedef struct camellia_key {
    unsigned int bits;
    uint32_t key[CAMELLIA_TABLE_WORD_LEN];
} CAMELLIA_KEY;

int CAMELLIA_set_key(const unsigned char *, const int, CAMELLIA_KEY *);

void CAMELLIA_encrypt(const unsigned char *, unsigned char *,
		      const CAMELLIA_KEY *);
void CAMELLIA_decrypt(const unsigned char *, unsigned char *,
		      const CAMELLIA_KEY *);

void CAMELLIA_cbc_encrypt(const unsigned char *, unsigned char *,
			  unsigned long, const CAMELLIA_KEY *,
			  unsigned char *, int);

#endif /* HEIM_CAMELLIA_H */
