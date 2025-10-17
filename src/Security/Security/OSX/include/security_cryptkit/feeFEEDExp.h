/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 18, 2024.
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
#ifndef	_CK_FEEFEEDEXP_H_
#define _CK_FEEFEEDEXP_H_

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
typedef void *feeFEEDExp;

/*
 * Alloc and init a feeFEEDExp object associated with specified feePubKey.
 */
feeFEEDExp feeFEEDExpNewWithPubKey(
	feePubKey pubKey,
	feeRandFcn randFcn,		// optional 
	void *randRef);

void feeFEEDExpFree(feeFEEDExp feed);

/*
 * Plaintext block size.
 */
unsigned feeFEEDExpPlainBlockSize(feeFEEDExp feed);

/*
 * Ciphertext block size used for decryption.
 */
unsigned feeFEEDExpCipherBlockSize(feeFEEDExp feed);

/*
 * Required size of buffer for ciphertext, upon encrypting one
 * block of plaintext.
 */
unsigned feeFEEDExpCipherBufSize(feeFEEDExp feed);

/*
 * Return the size of ciphertext to hold specified size of encrypted plaintext.
 */
unsigned feeFEEDExpCipherTextSize(feeFEEDExp feed, unsigned plainTextSize);

/*
 * Return the size of plaintext to hold specified size of decrypted ciphertext.
 */
unsigned feeFEEDExpPlainTextSize(feeFEEDExp feed, unsigned cipherTextSize);

/*
 * Encrypt a block or less of data. Caller malloc's cipherText. Generates
 * feeFEEDExpCipherBlockSize() bytes of cipherText if finalBlock is false;
 * if finalBlock is true it could produce twice as much ciphertext. 
 * If plainTextLen is less than feeFEEDExpPlainBlockSize(), finalBlock must be true.
 */
feeReturn feeFEEDExpEncryptBlock(feeFEEDExp feed,
	const unsigned char *plainText,
	unsigned plainTextLen,
	unsigned char *cipherText,
	unsigned *cipherTextLen,		// RETURNED
	int finalBlock);

/*
 * Decrypt (exactly) a block of data. Caller malloc's plainText. Always
 * generates feeFEEDExpBlockSize bytes of plainText, unless 'finalBlock' is
 * non-zero (in which case feeFEEDExpBlockSize or less bytes of plainText are
 * generated).
 */
feeReturn feeFEEDExpDecryptBlock(feeFEEDExp feed,
	const unsigned char *cipherText,
	unsigned cipherTextLen,
	unsigned char *plainText,
	unsigned *plainTextLen,			// RETURNED
	int finalBlock);

/*
 * Convenience routines to encrypt & decrypt multi-block data.
 */
feeReturn feeFEEDExpEncrypt(feeFEEDExp feed,
	const unsigned char *plainText,
	unsigned plainTextLen,
	unsigned char **cipherText,		// malloc'd and RETURNED
	unsigned *cipherTextLen);		// RETURNED

feeReturn feeFEEDExpDecrypt(feeFEEDExp feed,
	const unsigned char *cipherText,
	unsigned cipherTextLen,
	unsigned char **plainText,		// malloc'd and RETURNED
	unsigned *plainTextLen);		// RETURNED

#ifdef __cplusplus
}
#endif

#endif	/*_CK_FEEFEEDEXP_H_*/
