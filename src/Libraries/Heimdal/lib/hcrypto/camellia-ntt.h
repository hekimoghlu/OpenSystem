/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 18, 2023.
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
#ifndef HEADER_CAMELLIA_H
#define HEADER_CAMELLIA_H

#ifdef  __cplusplus
extern "C" {
#endif

#define CAMELLIA_BLOCK_SIZE 16
#define CAMELLIA_TABLE_BYTE_LEN 272
#define CAMELLIA_TABLE_WORD_LEN (CAMELLIA_TABLE_BYTE_LEN / 4)

/* u32 must be 32bit word */
typedef uint32_t u32;
typedef unsigned char u8;

typedef u32 KEY_TABLE_TYPE[CAMELLIA_TABLE_WORD_LEN];


void Camellia_Ekeygen(const int keyBitLength,
		      const unsigned char *rawKey,
		      KEY_TABLE_TYPE keyTable);

void Camellia_EncryptBlock(const int keyBitLength,
			   const unsigned char *plaintext,
			   const KEY_TABLE_TYPE keyTable,
			   unsigned char *cipherText);

void Camellia_DecryptBlock(const int keyBitLength,
			   const unsigned char *cipherText,
			   const KEY_TABLE_TYPE keyTable,
			   unsigned char *plaintext);


#ifdef  __cplusplus
}
#endif

#endif /* HEADER_CAMELLIA_H */
