/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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

#ifndef HEIM_AES_H
#define HEIM_AES_H 1

/* symbol renaming */
#define AES_set_encrypt_key hc_AES_set_encrypt_key
#define AES_set_decrypt_key hc_AES_decrypt_key
#define AES_encrypt hc_AES_encrypt
#define AES_decrypt hc_AES_decrypt
#define AES_cbc_encrypt hc_AES_cbc_encrypt
#define AES_cfb8_encrypt hc_AES_cfb8_encrypt

/*
 *
 */

#define AES_BLOCK_SIZE 16
#define AES_MAXNR 14

#define AES_ENCRYPT 1
#define AES_DECRYPT 0

typedef struct aes_key {
    uint32_t key[(AES_MAXNR+1)*4];
    int rounds;
} AES_KEY;

#ifdef __cplusplus
extern "C" {
#endif

int AES_set_encrypt_key(const unsigned char *, const int, AES_KEY *);
int AES_set_decrypt_key(const unsigned char *, const int, AES_KEY *);

void AES_encrypt(const unsigned char *, unsigned char *, const AES_KEY *);
void AES_decrypt(const unsigned char *, unsigned char *, const AES_KEY *);

void AES_cbc_encrypt(const unsigned char *, unsigned char *,
		     unsigned long, const AES_KEY *,
		     unsigned char *, int);
void AES_cfb8_encrypt(const unsigned char *, unsigned char *,
		      unsigned long, const AES_KEY *,
		      unsigned char *, int);


void AES_cfb8_encrypt(const unsigned char *, unsigned char *,
		      const unsigned long, const AES_KEY *,
		      unsigned char *, int);

#ifdef  __cplusplus
}
#endif

#endif /* HEIM_AES_H */
