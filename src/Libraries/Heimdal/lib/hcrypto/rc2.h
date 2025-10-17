/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 30, 2022.
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

/* symbol renaming */
#define RC2_set_key hc_RC2_set_key
#define RC2_encryptc hc_RC2_encryptc
#define RC2_decryptc hc_RC2_decryptc
#define RC2_cbc_encrypt hc_RC2_cbc_encrypt

/*
 *
 */

#define RC2_ENCRYPT	1
#define RC2_DECRYPT	0

#define RC2_BLOCK_SIZE	8
#define RC2_BLOCK	RC2_BLOCK_SIZE
#define RC2_KEY_LENGTH	16

typedef struct rc2_key {
    unsigned int data[64];
} RC2_KEY;

#ifdef __cplusplus
extern "C" {
#endif

void RC2_set_key(RC2_KEY *, int, const unsigned char *,int);

void RC2_encryptc(unsigned char *, unsigned char *, const RC2_KEY *);
void RC2_decryptc(unsigned char *, unsigned char *, const RC2_KEY *);

void RC2_cbc_encrypt(const unsigned char *, unsigned char *, long,
		     RC2_KEY *, unsigned char *, int);

#ifdef  __cplusplus
}
#endif
