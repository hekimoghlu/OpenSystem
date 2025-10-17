/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 28, 2024.
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
#ifndef	_CK_FEEFEED_H_
#define _CK_FEEFEED_H_

#if	!defined(__MACH__)
#include <ckconfig.h>
#include <feeTypes.h>
#include <feePublicKey.h>
#else
#include <security_cryptkit/ckconfig.h>
#include <security_cryptkit/feeTypes.h>
#include <security_cryptkit/feePublicKey.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Opaque object handle.
 */
typedef void *feeFEED;

/*
 * forEncrypt argument values.
 */
#define FF_DECRYPT	0
#define FF_ENCRYPT	1

/*
 * Alloc and init a feeFEED object associated with specified feePubKey
 * objects.
 */
feeFEED feeFEEDNewWithPubKey(feePubKey myPrivKey,
	feePubKey theirPubKey,
	int forEncrypt,			// FF_DECRYPT, FF_ENCRYPT
	feeRandFcn randFcn,		// optional 
	void *randRef);

void feeFEEDFree(feeFEED feed);

/*
 * Plaintext block size.
 */
unsigned feeFEEDPlainBlockSize(feeFEED feed);

/*
 * Ciphertext block size used for decryption.
 */
unsigned feeFEEDCipherBlockSize(feeFEED feed);

/*
 * Calculate size of buffer currently needed to encrypt one block of
 * plaintext.
 */
unsigned feeFEEDCipherBufSize(feeFEED feed,
	 int finalBlock);

/*
 * Return the size of ciphertext currently needed to encrypt specified 
 * size of plaintext. Also can be used to calculate size of ciphertext 
 * which can be decrypted into specified size of plaintext. 
 */
unsigned feeFEEDCipherTextSize(feeFEED feed, 
	unsigned plainTextSize,
	int finalBlock);

/*
 * Return the size of plaintext currently needed to decrypt specified size 
 * of ciphertext. Also can be used to calculate size of plaintext 
 * which can be encrypted into specified size of ciphertext.
 */
unsigned feeFEEDPlainTextSize(feeFEED feed, 
	unsigned 	cipherTextSize,
	int 		finalBlock);			// ignored if decrypting

/*
 * Encrypt a block or less of data. Caller malloc's cipherText.
 */
feeReturn feeFEEDEncryptBlock(feeFEED feed,
	const unsigned char *plainText,
	unsigned plainTextLen,
	unsigned char *cipherText,
	unsigned *cipherTextLen,		// RETURNED
	int finalBlock);

/*
 * Decrypt (exactly) a block of data. Caller malloc's plainText. Always
 * generates feeFEEDBlockSize bytes of plainText, unless 'finalBlock' is
 * non-zero (in which case feeFEEDBlockSize or less bytes of plainText are
 * generated).
 */
feeReturn feeFEEDDecryptBlock(feeFEED feed,
	const unsigned char *cipherText,
	unsigned cipherTextLen,
	unsigned char *plainText,
	unsigned *plainTextLen,			// RETURNED
	int finalBlock);

/*
 * Convenience routines to encrypt & decrypt multi-block data.
 */
feeReturn feeFEEDEncrypt(feeFEED feed,
	const unsigned char *plainText,
	unsigned plainTextLen,
	unsigned char **cipherText,		// malloc'd and RETURNED
	unsigned *cipherTextLen);		// RETURNED

feeReturn feeFEEDDecrypt(feeFEED feed,
	const unsigned char *cipherText,
	unsigned cipherTextLen,
	unsigned char **plainText,		// malloc'd and RETURNED
	unsigned *plainTextLen);		// RETURNED

#ifdef __cplusplus
}
#endif

#endif	/*_CK_FEEFEED_H_*/
