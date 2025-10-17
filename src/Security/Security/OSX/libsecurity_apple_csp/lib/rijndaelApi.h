/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 23, 2024.
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
/* 
 * rijndaelApi.h  -  AES API layer
 *
 * Based on rijndael-api-ref.h v2.0 written by Paulo Barreto
 * and Vincent Rijmen
 */

#ifndef	_RIJNDAEL_API_REF_H_
#define _RIJNDAEL_API_REF_H_

#include <stdio.h>
#include "rijndael-alg-ref.h"

#ifdef	__cplusplus
extern "C" {
#endif

/*  Error Codes  */
#define     BAD_KEY_MAT        -1  /*  Key material not of correct 
									   length */
#define     BAD_KEY_INSTANCE   -2  /*  Key passed is not valid  */

#define     MAX_AES_KEY_SIZE	(MAX_AES_KEY_BITS / 8)
#define 	MAX_AES_BLOCK_SIZE	(MAX_AES_BLOCK_BITS / 8)
#define     MAX_AES_IV_SIZE		MAX_AES_BLOCK_SIZE
	
#define		TRUE		1
#define 	FALSE		0

/*  The structure for key information */
typedef struct {
	word32   		keyLen;		/* Length of the key in bits */
	word32  		blockLen;   /* Length of block in bits */
	word32			columns;	/* optimization, blockLen / 32 */
	word8 			keySched[MAXROUNDS+1][4][MAXBC];	
} keyInstance;

int makeKey(
	keyInstance *key, 
	int keyLen, 		// in BITS
	int blockLen,		// in BITS
	word8 *keyMaterial,
	int enable128Opt);

/*
 * Simplified single-block encrypt/decrypt.
 */
int rijndaelBlockEncrypt(
	keyInstance *key, 
	word8 *input, 
	word8 *outBuffer);
int rijndaelBlockDecrypt(
	keyInstance *key, 
	word8 *input, 
	word8 *outBuffer);
	
#if		!GLADMAN_AES_128_ENABLE
/*
 * Optimized routines for 128 bit block and 128 bit key.
 */
int rijndaelBlockEncrypt128(
	keyInstance 	*key, 
	word8 			*input, 
	word8 			*outBuffer);
int rijndaelBlockDecrypt128(
	keyInstance 	*key, 
	word8 			*input, 
	word8 			*outBuffer);
#endif	/* !GLADMAN_AES_128_ENABLE */

#if defined(__ppc__) && defined(ALTIVEC_ENABLE)
/* 
 * dmitch addenda 4/11/2001: 128-bit only vectorized encrypt/decrypt with no CBC
 */
void vBlockEncrypt128(
	keyInstance *key, 
	word8 *input, 
	word8 *outBuffer);
void vBlockDecrypt128(
	keyInstance *key, 
	word8 *input, 
	word8 *outBuffer);

/* temp switch for runtime enable/disable */
extern int doAES128;

#endif	/* __ppc__ && ALTIVEC_ENABLE */
	
/* ptr to one of several (possibly optimized) encrypt/decrypt functions */
typedef int (*aesCryptFcn)(
	keyInstance *key, 
	word8 *input, 
	word8 *outBuffer);

#ifdef	__cplusplus
}
#endif	// cplusplus

#endif	// RIJNDAEL_API_REF


