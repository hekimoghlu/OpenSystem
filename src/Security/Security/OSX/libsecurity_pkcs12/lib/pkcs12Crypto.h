/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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
 * p12Crypto.h - PKCS12 Crypto routines.
 */
 
#ifndef	_PKCS12_CRYPTO_H_
#define _PKCS12_CRYPTO_H_

#include <Security/Security.h>
#include <security_asn1/SecNssCoder.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Given appropriate P12-style parameters, cook up a CSSM_KEY.
 */
CSSM_RETURN p12KeyGen(
	CSSM_CSP_HANDLE		cspHand,
	CSSM_KEY			&key,
	bool				isForEncr,	// true: en/decrypt   false: MAC
	CSSM_ALGORITHMS		keyAlg,
	CSSM_ALGORITHMS		pbeHashAlg,	// SHA1, MD5 only
	uint32				keySizeInBits,
	uint32				iterCount,
	const CSSM_DATA		&salt,
	
	/* exactly one of the following two must be valid */
	const CSSM_DATA		*pwd,		// unicode, double null terminated
	const CSSM_KEY		*passKey,
	CSSM_DATA			&iv);		// referent is optional
	
/*
 * Decrypt (typically, an encrypted P7 ContentInfo contents or
 * a P12 ShroudedKeyBag).
 */
CSSM_RETURN p12Decrypt(
	CSSM_CSP_HANDLE		cspHand,
	const CSSM_DATA		&cipherText,
	CSSM_ALGORITHMS		keyAlg,				
	CSSM_ALGORITHMS		encrAlg,
	CSSM_ALGORITHMS		pbeHashAlg,			// SHA1, MD5 only
	uint32				keySizeInBits,
	uint32				blockSizeInBytes,	// for IV
	CSSM_PADDING		padding,			// CSSM_PADDING_PKCS7, etc.
	CSSM_ENCRYPT_MODE	mode,				// CSSM_ALGMODE_CBCPadIV8, etc.
	uint32				iterCount,
	const CSSM_DATA		&salt,
	/* exactly one of the following two must be valid */
	const CSSM_DATA		*pwd,		// unicode, double null terminated
	const CSSM_KEY		*passKey,
	SecNssCoder			&coder,		// for mallocing KeyData and plainText
	CSSM_DATA			&plainText);

/*
 * Decrypt (typically, an encrypted P7 ContentInfo contents)
 */
CSSM_RETURN p12Encrypt(
	CSSM_CSP_HANDLE		cspHand,
	const CSSM_DATA		&plainText,
	CSSM_ALGORITHMS		keyAlg,				
	CSSM_ALGORITHMS		encrAlg,
	CSSM_ALGORITHMS		pbeHashAlg,			// SHA1, MD5 only
	uint32				keySizeInBits,
	uint32				blockSizeInBytes,	// for IV
	CSSM_PADDING		padding,			// CSSM_PADDING_PKCS7, etc.
	CSSM_ENCRYPT_MODE	mode,				// CSSM_ALGMODE_CBCPadIV8, etc.
	uint32				iterCount,
	const CSSM_DATA		&salt,
	const CSSM_DATA		*pwd,		// unicode, double null terminated
	const CSSM_KEY		*passKey,
	SecNssCoder			&coder,		// for mallocing cipherText
	CSSM_DATA			&cipherText);

/*
 * Calculate the MAC for a PFX. Caller is either going compare
 * the result against an existing PFX's MAC or drop the result into 
 * a newly created PFX.
 */
CSSM_RETURN p12GenMac(
	CSSM_CSP_HANDLE		cspHand,
	const CSSM_DATA		&ptext,	// e.g., NSS_P12_DecodedPFX.derAuthSaafe
	CSSM_ALGORITHMS		alg,	// better be SHA1!
	unsigned			iterCount,
	const CSSM_DATA		&salt,
	/* exactly one of the following two must be valid */
	const CSSM_DATA		*pwd,		// unicode, double null terminated
	const CSSM_KEY		*passKey,
	SecNssCoder			&coder,		// for mallocing macData
	CSSM_DATA			&macData);	// RETURNED 

/*
 * Unwrap a shrouded key.
 */
CSSM_RETURN p12UnwrapKey(
	CSSM_CSP_HANDLE		cspHand,
	CSSM_DL_DB_HANDLE_PTR	dlDbHand,		// optional
	int					keyIsPermanent,		// nonzero - store in DB
	const CSSM_DATA		&shroudedKeyBits,
	CSSM_ALGORITHMS		keyAlg,				// of the unwrapping key
	CSSM_ALGORITHMS		encrAlg,
	CSSM_ALGORITHMS		pbeHashAlg,			// SHA1, MD5 only
	uint32				keySizeInBits,
	uint32				blockSizeInBytes,	// for IV
	CSSM_PADDING		padding,			// CSSM_PADDING_PKCS7, etc.
	CSSM_ENCRYPT_MODE	mode,				// CSSM_ALGMODE_CBCPadIV8, etc.
	uint32				iterCount,
	const CSSM_DATA		&salt,
	/* exactly one of the following two must be valid */
	const CSSM_DATA		*pwd,		// unicode, double null terminated
	const CSSM_KEY		*passKey,
	SecNssCoder			&coder,		// for mallocing privKey
	const CSSM_DATA		&labelData,
	SecAccessRef		access,		// optional; use default ACL if NULL and !noAcl
	bool				noAcl,		// true ==> no ACL
	CSSM_KEYUSE			keyUsage,
	CSSM_KEYATTR_FLAGS	keyAttrs,

	/*
	 * Result: a private key, reference format, optionaly stored
	 * in dlDbHand
	 */
	CSSM_KEY_PTR		&privKey);

CSSM_RETURN p12WrapKey(
	CSSM_CSP_HANDLE		cspHand,
	CSSM_KEY_PTR		privKey,
	const CSSM_ACCESS_CREDENTIALS *privKeyCreds,
	CSSM_ALGORITHMS		keyAlg,				// of the unwrapping key
	CSSM_ALGORITHMS		encrAlg,
	CSSM_ALGORITHMS		pbeHashAlg,			// SHA1, MD5 only
	uint32				keySizeInBits,
	uint32				blockSizeInBytes,	// for IV
	CSSM_PADDING		padding,			// CSSM_PADDING_PKCS7, etc.
	CSSM_ENCRYPT_MODE	mode,				// CSSM_ALGMODE_CBCPadIV8, etc.
	uint32				iterCount,
	const CSSM_DATA		&salt,
	/* exactly one of the following two must be valid */
	const CSSM_DATA		*pwd,		// unicode, double null terminated
	const CSSM_KEY		*passKey,
	SecNssCoder			&coder,		// for mallocing keyBits
	CSSM_DATA			&shroudedKeyBits);	// RETURNED

#ifdef __cplusplus
}
#endif

#endif	/* _PKCS12_CRYPTO_H_ */

