/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 23, 2024.
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
 * opensslAsn1.h - ANS1 encode/decode of openssl object, libssnasn1 version
 */
 
#ifndef	_OPENSSL_ASN1_H_
#define _OPENSSL_ASN1_H_


#include <openssl/rsa_legacy.h>
#include <openssl/dsa_legacy.h>
#include <openssl/dh_legacy.h>
#include <Security/osKeyTemplates.h>
#include <Security/cssmtype.h>
#include <security_cdsa_utilities/cssmdata.h>
#include <security_asn1/SecNssCoder.h>

#ifdef	__cplusplus
extern "C" {
#endif

/* CSSM_DATA --> BIGNUM */
BIGNUM *cssmDataToBn(
	const CSSM_DATA &cdata);
	
/* BIGNUM --> CSSM_DATA, mallocing from a SecNssCoder's PL_ArenaPool */
void bnToCssmData(
	const BIGNUM *bn,
	CSSM_DATA &cdata,
	SecNssCoder &coder);

/* CSSM_DATA --> unsigned int */
unsigned cssmDataToInt(
	const CSSM_DATA &cdata);

/* unsigned int --> CSSM_DATA, mallocing from an SecNssCoder */
void intToCssmData(
	unsigned num,
	CSSM_DATA &cdata,
	SecNssCoder &coder);

/*
 * DER encode/decode RSA keys in various formats. 
 */
CSSM_RETURN RSAPublicKeyDecode(
	RSA 				*openKey, 
	CSSM_KEYBLOB_FORMAT	format,
	void 				*p, 
	size_t				length);
CSSM_RETURN	RSAPublicKeyEncode(
	RSA 				*openKey, 
	CSSM_KEYBLOB_FORMAT	format,
	const CssmData		&descData,
	CssmOwnedData		&encodedKey);
CSSM_RETURN RSAPrivateKeyDecode(
	RSA 				*openKey, 
	CSSM_KEYBLOB_FORMAT	format,
	void 				*p, 
	size_t				length);
CSSM_RETURN	RSAPrivateKeyEncode(
	RSA 				*openKey, 
	CSSM_KEYBLOB_FORMAT	format,
	const CssmData		&descData,
	CssmOwnedData		&encodedKey);
CSSM_RETURN RSAOAEPPublicKeyDecode(
	RSA 				*openKey, 
	void 				*p, 
	size_t				length,
	/* mallocd and returned label */
	CSSM_DATA			*label);
CSSM_RETURN	RSAOAEPPublicKeyEncode(
	RSA 				*openKey, 
	const CSSM_DATA		*label,
	CssmOwnedData		&encodedKey);
CSSM_RETURN RSAOAEPPrivateKeyDecode(
	RSA 				*openKey, 
	void 				*p, 
	size_t				length,
	/* mallocd and returned label */
	CSSM_DATA			*label);
CSSM_RETURN	RSAOAEPPrivateKeyEncode(
	RSA 				*openKey, 
	const CSSM_DATA		*label,
	CssmOwnedData		&encodedKey);

CSSM_RETURN generateDigestInfo(
	const void		*messageDigest,
	size_t			digestLen,
	CSSM_ALGORITHMS	digestAlg,		// CSSM_ALGID_SHA1, etc.
	CssmOwnedData	&encodedInfo,
	size_t			maxEncodedSize);
CSSM_RETURN DSAPublicKeyDecode(
	DSA 				*openKey, 
	CSSM_KEYBLOB_FORMAT	format,
	void 				*p, 
	size_t				length);
CSSM_RETURN	DSAPublicKeyEncode(
	DSA 				*openKey, 
	CSSM_KEYBLOB_FORMAT	format,
	const CssmData		&descData,
	CssmOwnedData		&encodedKey);
CSSM_RETURN DSAPrivateKeyDecode(
	DSA 				*openKey, 
	CSSM_KEYBLOB_FORMAT	format,
	void 				*p, 
	size_t	 			length);
CSSM_RETURN	DSAPrivateKeyEncode(
	DSA 				*openKey, 
	CSSM_KEYBLOB_FORMAT	format,
	const CssmData		&descData,
	CssmOwnedData		&encodedKey);

CSSM_RETURN DSASigEncode(
	DSA_SIG			*openSig,
	CssmOwnedData	&encodedSig);
CSSM_RETURN DSASigDecode(
	DSA_SIG 		*openSig, 
	const void 		*p, 
	unsigned		length);

CSSM_RETURN DSAEncodeAlgParams(
	NSS_DSAAlgParams	&algParams,
	CssmOwnedData		&encodedParams);
CSSM_RETURN DSADecodeAlgParams(
	NSS_DSAAlgParams	&algParams,
	const void			*p,
	unsigned			len,
	SecNssCoder			&coder);
	
CSSM_RETURN DHPrivateKeyDecode(
	DH	 				*openKey, 
	CSSM_KEYBLOB_FORMAT	format,
	unsigned char 		*p, 
	unsigned 			length);
CSSM_RETURN	DHPrivateKeyEncode(
	DH	 				*openKey, 
	CSSM_KEYBLOB_FORMAT	format,
	CssmOwnedData		&encodedKey);
CSSM_RETURN DHPublicKeyDecode(
	DH	 				*openKey, 
	CSSM_KEYBLOB_FORMAT	format,
	unsigned char 		*p, 
	unsigned 			length);
CSSM_RETURN	DHPublicKeyEncode(
	DH	 				*openKey, 
	CSSM_KEYBLOB_FORMAT	format,
	CssmOwnedData		&encodedKey);
CSSM_RETURN DHParamBlockDecode(
	const CSSM_DATA &encParam,
	NSS_DHParameterBlock &paramBlock,
	SecNssCoder &coder);

CSSM_RETURN generateDigestInfo(
	const void		*msgDigest,
	size_t			digestLen,
	CSSM_ALGORITHMS	digestAlg,		// CSSM_ALGID_SHA1, etc.
	CssmOwnedData	&encodedInfo,
	size_t			maxEncodedSize);

#ifdef	__cplusplus
}
#endif

#endif	/* _OPENSSL_ASN1_H_ */
