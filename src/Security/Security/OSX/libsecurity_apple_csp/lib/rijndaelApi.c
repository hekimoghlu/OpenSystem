/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 4, 2021.
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
 * rijndaelApi.c  -  AES API layer
 *
 * Based on rijndael-api-ref.h v2.0 written by Paulo Barreto
 * and Vincent Rijmen
 */
#include <stdlib.h>
#include <string.h>

#include "rijndael-alg-ref.h"
#include "rijndaelApi.h"

#ifdef	ALTIVEC_ENABLE
/* this goes somewhere else and gets init'd by the plugin object.... */
/* as of 4/11/2001, the vectorized routines do NOT work */
int gHasAltivec = 0;
#endif

int doAES128 = 1;

#define CBC_DEBUG		0
#if 	CBC_DEBUG
static void dumpChainBuf(cipherInstance *cipher, char *op)
{
	int t,j;
	int columns = cipher->blockLen / 32;

	printf("chainBuf %s: ", op);
	for (j = 0; j < columns; j++) {
		for(t = 0; t < 4; t++) {
			printf("%02x ", cipher->chainBlock[t][j]);
		}
	}
	printf("\n");
}
#else
#define dumpChainBuf(c, o)
#endif

int makeKey(	
	keyInstance *key, 
	int keyLen, 		// in BITS
	int blockLen,		// in BITS
	word8 *keyMaterial,
	int enable128Opt)
{
	unsigned keyBytes;
	unsigned  i;

	if (key == NULL) {
		return BAD_KEY_INSTANCE;
	}
	if(keyMaterial == NULL) {
		return BAD_KEY_MAT;
	}
	if ((keyLen == 128) || (keyLen == 192) || (keyLen == 256)) { 
		key->keyLen = keyLen;
	} else {
		return BAD_KEY_MAT;
	}
	key->blockLen = blockLen;
	key->columns = blockLen / 32;
	
	/* initialize key schedule */ 
#if		!GLADMAN_AES_128_ENABLE
	if(enable128Opt && 
			(keyLen == MIN_AES_KEY_BITS) && 
			(blockLen == MIN_AES_BLOCK_BITS)) {
		/* optimized, 128 bit key and block size */
		word8 k[4][KC_128_OPT] __attribute__((aligned(4)));
		
		for(i = 0; i < (MIN_AES_KEY_BITS/8); i++) {
			/* speed this up */
			k[i % 4][i / 4] = keyMaterial[i]; 
		}	
		rijndaelKeySched128 (k, key->keySched);	
		memset(k, 0, 4 * KC_128_OPT);
	}
	else 
#endif	/* !GLADMAN_AES_128_ENABLE */
	{

		/* general case */
		word8 k[4][MAXKC];

		keyBytes = keyLen / 8;
		for(i = 0; i < keyBytes; i++) {
			k[i % 4][i / 4] = keyMaterial[i]; 
		}	
		rijndaelKeySched (k, key->keyLen, key->blockLen, key->keySched);	
		memset(k, 0, 4 * MAXKC);
	}
	return TRUE;
}

/*
 * Simplified single-block encrypt/decrypt.
 */
#define AES_CONSISTENCY_CHECK		1

int rijndaelBlockEncrypt(
	keyInstance *key, 
	word8 *input, 
	word8 *outBuffer)
{
	int t;
	unsigned j;
	word8 localBlock[4][MAXBC];		// working memory: encrypt/decrypt in place here
	
	#if		AES_CONSISTENCY_CHECK
	if (key == NULL ||
		(key->keyLen != 128 && key->keyLen != 192 && key->keyLen != 256) ||
		(key->blockLen != 128 && key->blockLen != 192 && key->blockLen != 256)) {
		return BAD_KEY_INSTANCE;
	}
	#endif	/* AES_CONSISTENCY_CHECK */
	
	#if	defined(__ppc__) && defined(ALTIVEC_ENABLE)
	if(gHasAltivec && (key->blockLen == 128)) {
		vBlockEncrypt128(key, input, outBuffer);
		return 128; 
	}
	#endif
	
	for (j = 0; j < key->columns; j++) {
		for(t = 0; t < 4; t++)
		/* parse input stream into rectangular array */
			localBlock[t][j] = input[4*j+t];
	}
	rijndaelEncrypt (localBlock, key->keyLen, key->blockLen, key->keySched);
	for (j = 0; j < key->columns; j++) {
		/* parse rectangular array into output ciphertext bytes */
		for(t = 0; t < 4; t++)
			outBuffer[4*j+t] = (word8) localBlock[t][j];
	}
	memset(localBlock, 0, 4 * MAXBC);
	return key->blockLen;
}

int rijndaelBlockDecrypt(
	keyInstance *key, 
	word8 *input, 
	word8 *outBuffer)
{
	int t;
	unsigned j;
	word8 localBlock[4][MAXBC];		// working memory: encrypt/decrypt in place here

	#if		AES_CONSISTENCY_CHECK
	if (key == NULL ||
		(key->keyLen != 128 && key->keyLen != 192 && key->keyLen != 256) ||
		(key->blockLen != 128 && key->blockLen != 192 && key->blockLen != 256)) {
		return BAD_KEY_INSTANCE;
	}
	#endif	/* AES_CONSISTENCY_CHECK */
	
	#if	defined(__ppc__) && defined(ALTIVEC_ENABLE)
	if(gHasAltivec && (cipher->blockLen == 128)) {
		vBlockDecrypt128(key, input, outBuffer);
		return 128;
	}
	#endif
	
	for (j = 0; j < key->columns; j++) {
		for(t = 0; t < 4; t++)
		/* parse input stream into rectangular array */
			localBlock[t][j] = input[4*j+t];
	}
	rijndaelDecrypt (localBlock, key->keyLen, key->blockLen, key->keySched);
	for (j = 0; j < key->columns; j++) {
		/* parse rectangular array into output ciphertext bytes */
		for(t = 0; t < 4; t++)
			outBuffer[4*j+t] = (word8) localBlock[t][j];
	}
	memset(localBlock, 0, 4 * MAXBC);
	return key->blockLen;
}

#if		!GLADMAN_AES_128_ENABLE
/*
 * Optimized routines for 128 bit block and 128 bit key.
 */
int rijndaelBlockEncrypt128(
	keyInstance *key, 
	word8 *input, 
	word8 *outBuffer)
{
	int j;
	word8 localBlock[4][BC_128_OPT] __attribute__((aligned(4)));
	word8 *row0 = localBlock[0];
	word8 *row1 = localBlock[1];
	word8 *row2 = localBlock[2];
	word8 *row3 = localBlock[3];
	
	/* parse input stream into rectangular array */
	for (j = 0; j < BC_128_OPT; j++) {
		*row0++ = *input++;
		*row1++ = *input++;
		*row2++ = *input++;
		*row3++ = *input++;
	}
	rijndaelEncrypt128 (localBlock, key->keySched);
	
	/* parse rectangular array into output ciphertext bytes */
	row0 = localBlock[0];
	row1 = localBlock[1];
	row2 = localBlock[2];
	row3 = localBlock[3];

	for (j = 0; j < BC_128_OPT; j++) {
		*outBuffer++ = *row0++;
		*outBuffer++ = *row1++;
		*outBuffer++ = *row2++;
		*outBuffer++ = *row3++;
	}
	memset(localBlock, 0, 4*BC_128_OPT);
	return MIN_AES_BLOCK_BITS;
}

int rijndaelBlockDecrypt128(
	keyInstance *key, 
	word8 *input, 
	word8 *outBuffer)
{
	int j;
	word8 localBlock[4][BC_128_OPT] __attribute__((aligned(4)));	
	word8 *row0 = localBlock[0];
	word8 *row1 = localBlock[1];
	word8 *row2 = localBlock[2];
	word8 *row3 = localBlock[3];
	
	/* parse input stream into rectangular array */
	for (j = 0; j < BC_128_OPT; j++) {
		*row0++ = *input++;
		*row1++ = *input++;
		*row2++ = *input++;
		*row3++ = *input++;
	}
	
	rijndaelDecrypt128 (localBlock, key->keySched);
	
	/* parse rectangular array into output ciphertext bytes */
	row0 = localBlock[0];
	row1 = localBlock[1];
	row2 = localBlock[2];
	row3 = localBlock[3];

	for (j = 0; j < BC_128_OPT; j++) {
		*outBuffer++ = *row0++;
		*outBuffer++ = *row1++;
		*outBuffer++ = *row2++;
		*outBuffer++ = *row3++;
	}
	memset(localBlock, 0, 4*BC_128_OPT);
	return MIN_AES_BLOCK_BITS;
}
#endif		/* !GLADMAN_AES_128_ENABLE */

