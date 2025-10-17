/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 28, 2023.
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
 * opensshCoding.h - Encoding and decoding of OpenSSH format public keys.
 *
 */

#ifndef	_OPENSSH_CODING_H_
#define _OPENSSH_CODING_H_

#include <openssl/rsa_legacy.h>
#include <openssl/dsa_legacy.h>
#include <Security/cssmtype.h>
#include <security_cdsa_utilities/cssmdata.h>
#include <CoreFoundation/CFData.h>

#ifdef	__cplusplus
extern "C" {
#endif

void appendUint32OpenSSH(
	CFMutableDataRef cfOut,
	uint32_t ui);
uint32_t readUint32OpenSSH(
	const unsigned char *&cp,		// IN/OUT
	unsigned &len);					// IN/OUT 

extern CSSM_RETURN RSAPublicKeyEncodeOpenSSH1(
	RSA 			*openKey, 
	const CssmData	&descData,
	CssmOwnedData	&encodedKey);

extern CSSM_RETURN RSAPublicKeyDecodeOpenSSH1(
	RSA 			*openKey, 
	void 			*p, 
	size_t			length);

extern CSSM_RETURN RSAPrivateKeyEncodeOpenSSH1(
	RSA 			*openKey, 
	const CssmData	&descData,
	CssmOwnedData	&encodedKey);

extern CSSM_RETURN RSAPrivateKeyDecodeOpenSSH1(
	RSA 			*openKey, 
	void 			*p, 
	size_t			length);

extern CSSM_RETURN RSAPublicKeyEncodeOpenSSH2(
	RSA 			*openKey, 
	const CssmData	&descData,
	CssmOwnedData	&encodedKey);

extern CSSM_RETURN RSAPublicKeyDecodeOpenSSH2(
	RSA 			*openKey, 
	void 			*p, 
	size_t			length);

extern CSSM_RETURN DSAPublicKeyEncodeOpenSSH2(
	DSA 			*openKey, 
	const CssmData	&descData,
	CssmOwnedData	&encodedKey);

extern CSSM_RETURN DSAPublicKeyDecodeOpenSSH2(
	DSA 			*openKey, 
	void 			*p, 
	size_t			length);

/* In opensshWrap.cpp */

/* Encode OpenSSHv1 private key, with or without encryption */
extern CSSM_RETURN encodeOpenSSHv1PrivKey(
	RSA					*r,
	const uint8			*comment,		/* optional */
	unsigned			commentLen,
	const uint8			*encryptKey,	/* optional; if present, it's 16 bytes of MD5(password) */
	CFDataRef			*encodedKey);	/* RETURNED */

extern CSSM_RETURN decodeOpenSSHv1PrivKey(
	const unsigned char *encodedKey,
	unsigned			encodedKeyLen,
	RSA					*r,
	const uint8			*decryptKey,	/* optional; if present, it's 16 bytes of MD5(password) */
	uint8				**comment,		/* mallocd and RETURNED */
	unsigned			*commentLen);	/* RETURNED */

#ifdef	__cplusplus
}
#endif

#endif	/* _OPENSSH_CODING_H_ */
