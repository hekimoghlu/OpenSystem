/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 17, 2025.
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
 * pkcs12Utils.h
 */
 
#ifndef	_PKCS12_UTILS_H_
#define _PKCS12_UTILS_H_

#include <Security/cssmtype.h>
#include <security_asn1/SecNssCoder.h>
#include <security_pkcs12/pkcs7Templates.h>
#include <security_pkcs12/pkcs12Templates.h>
#include <Security/cssmerr.h>
#include <CoreFoundation/CoreFoundation.h>

#ifdef __cplusplus
extern "C" {
#endif

/* malloc a NULL-ed array of pointers of size num+1 */
void **p12NssNullArray(
	uint32 num,
	SecNssCoder &coder);

/* CSSM_DATA --> uint32. Returns true if OK. */
bool p12DataToInt(
	const CSSM_DATA &cdata,
	uint32 &u);

/* uint32 --> CSSM_DATA */
void p12IntToData(
	uint32 num,
	CSSM_DATA &cdata,
	SecNssCoder &coder);

/* CFDataRef <--> CSSM_DATA */
CFDataRef p12CssmDataToCf(
	const CSSM_DATA &c) CF_RETURNS_RETAINED;
void p12CfDataToCssm(
	CFDataRef cf,
	CSSM_DATA &c,
	SecNssCoder &coder);

CSSM_DATA_PTR p12StringToUtf8(
	CFStringRef cfStr,
	SecNssCoder &coder);

const char *p12BagTypeStr(
	NSS_P12_SB_Type type);
const char *p7ContentInfoTypeStr(
	NSS_P7_CI_Type type);

/* map an OID to the components */

typedef enum {
	PW_None,			/* not comprehended */
	PW_PKCS5_v1_5,		/* PKCS5 v1.5 */
	PW_PKCS5_v2,		/* PKCS5 v2.0, not used by this module but parsed here */
	PW_PKCS12			/* PKCS12 */
} PKCS_Which;

/* returns false if OID not found */
bool pkcsOidToParams(
	const CSSM_OID 		*oid,
	CSSM_ALGORITHMS		&keyAlg,		// e.g., CSSM_ALGID_DES
	CSSM_ALGORITHMS		&encrAlg,		// e.g., CSSM_ALGID_3DES_3KEY_EDE
	CSSM_ALGORITHMS		&pbeHashAlg,	// SHA1 or MD5
	uint32				&keySizeInBits,
	uint32				&blockSizeInBytes,	// for IV, optional
	CSSM_PADDING		&padding,		// CSSM_PADDING_PKCS7, etc.
	CSSM_ENCRYPT_MODE	&mode,			// CSSM_ALGMODE_CBCPadIV8, etc.
	PKCS_Which			&pkcs);			// PW_PKCS5_v1_5 or PW_PKCS12

CSSM_RETURN p12VerifyMac(
	const NSS_P12_DecodedPFX 	&pfx,
	CSSM_CSP_HANDLE				cspHand,
	const CSSM_DATA				*pwd,	// unicode, double null terminated
	const CSSM_KEY				*passKey,
	SecNssCoder					&coder);// for temp mallocs

void p12GenSalt(
	CSSM_DATA 	&salt,
	SecNssCoder &coder);

void p12GenLabel(
	CSSM_DATA &label,
	SecNssCoder &coder);

void p12NullAlgParams(
	CSSM_X509_ALGORITHM_IDENTIFIER &algId);

/*
 * Free memory via specified plugin's app-level allocator
 */
void freeCssmMemory(
	CSSM_HANDLE	hand,
	void 		*p);

/*
 * Though it pains me to do this, I must. We "happen to know" the 
 * names (in string form) of two of a key's attributes. These
 * have not been published anywhere, they are hard-coded into 
 * the script (KeySchema.m4) which generates the KeySchema
 * tables. 
 */

/*
 * This one is initially the same as the "label" argument passed 
 * in to the CSP when creating or importing keys; it eventually
 * gets munged into the hash of the associated public key (
 * in our case, by p12SetPubKeyHash()).
 */
#define P12_KEY_ATTR_LABEL_AND_HASH		"Label"

/*
 * This one is the user-friendly name.
 */
#define P12_KEY_ATTR_PRINT_NAME			"PrintName"

/*
 * Find private key by label, modify its Label attr to be the
 * hash of the associated public key. 
 */
CSSM_RETURN p12SetPubKeyHash(
	CSSM_CSP_HANDLE 	cspHand,		// where the key lives
	CSSM_DL_DB_HANDLE 	dlDbHand,		// ditto
	CSSM_DATA			&keyLabel,		// for DB lookup
	CSSM_DATA_PTR		newPrintName,	// optional
	SecNssCoder			&coder,			// for mallocing newLabel
	CSSM_DATA			&newLabel,		// RETURNED with label as hash
	CSSM_KEY_PTR		&foundKey);		// RETURNED on dup key detect

CSSM_RETURN p12AddContextAttribute(CSSM_CC_HANDLE CCHandle,
	uint32 				AttributeType,
	uint32 				AttributeLength,
	const void 			*AttributePtr);

/*
 * Find private key by specified label, delete it.
 */
CSSM_RETURN p12DeleteKey(
	CSSM_DL_DB_HANDLE 	dlDbHand, 
	const CSSM_DATA		&keyLabel);

/* convert App passphrase to array of chars used in P12 PBE */
void p12ImportPassPhrase(
	CFStringRef			inPhrase,
	SecNssCoder			&coder,
	CSSM_DATA			&outPhrase);

/*
 * Standard error throwMes.
 * P12_ENCODE_ERR only occurs on DER-encode which should never fail. 
 */
#define P12_DECODE_ERR		errSecUnknownFormat
#define P12_ENCODE_ERR		errSecInternalComponent
#define P12_THROW_DECODE	MacOSError::throwMe(P12_DECODE_ERR)
#define P12_THROW_ENCODE	MacOSError::throwMe(P12_ENCODE_ERR)

#ifdef __cplusplus
}
#endif

#endif	/* _PKCS12_UTILS_H_ */

