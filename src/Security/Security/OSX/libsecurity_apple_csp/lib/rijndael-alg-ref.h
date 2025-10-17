/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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
/* rijndael-alg-ref.h   v2.0   August '99
 * Reference ANSI C code
 * authors: Paulo Barreto
 *          Vincent Rijmen
 */
#ifndef __RIJNDAEL_ALG_H
#define __RIJNDAEL_ALG_H

#include "aesCommon.h"

#define MAXBC				(MAX_AES_BLOCK_BITS/32)
#define MAXKC				(MAX_AES_KEY_BITS/32)
#define MAXROUNDS			14

#ifdef	__cplusplus
extern "C" {
#endif

typedef unsigned char		word8;	
typedef unsigned short		word16;	
typedef unsigned int		word32;


int rijndaelKeySched (word8 k[4][MAXKC], int keyBits, int blockBits, 
		word8 rk[MAXROUNDS+1][4][MAXBC]);
int rijndaelEncrypt (word8 a[4][MAXBC], int keyBits, int blockBits, 
		word8 rk[MAXROUNDS+1][4][MAXBC]);
#ifndef	__APPLE__
int rijndaelEncryptRound (word8 a[4][MAXBC], int keyBits, int blockBits, 
		word8 rk[MAXROUNDS+1][4][MAXBC], int rounds);
#endif
int rijndaelDecrypt (word8 a[4][MAXBC], int keyBits, int blockBits, 
		word8 rk[MAXROUNDS+1][4][MAXBC]);
#ifndef	__APPLE__
int rijndaelDecryptRound (word8 a[4][MAXBC], int keyBits, int blockBits, 
		word8 rk[MAXROUNDS+1][4][MAXBC], int rounds);
#endif

#if		!GLADMAN_AES_128_ENABLE

/*
 * Optimized routines for 128-bit block and key.
 */
#define ROUNDS_128_OPT	10
#define BC_128_OPT		4
#define KC_128_OPT		4

/*
 * These require 32-bit word-aligned a, k, and rk arrays 
 */
int rijndaelKeySched128 (word8 k[4][KC_128_OPT], 
	word8 rk[MAXROUNDS+1][4][MAXBC]);
int rijndaelEncrypt128 (word8 a[4][BC_128_OPT], 
	word8 rk[MAXROUNDS+1][4][MAXBC]);
int rijndaelDecrypt128 (word8 a[4][BC_128_OPT], 
	word8 rk[MAXROUNDS+1][4][MAXBC]);

#endif		/* !GLADMAN_AES_128_ENABLE */

#ifdef	__cplusplus
}
#endif

#endif /* __RIJNDAEL_ALG_H */
