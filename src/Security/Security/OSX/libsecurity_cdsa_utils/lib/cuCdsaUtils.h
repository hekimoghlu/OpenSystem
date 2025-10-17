/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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
	File:		 cuCdsaUtils.h
	
	Description: common CDSA access utilities

	Author:		 dmitch
*/

#ifndef	_COMMON_CDSA_UTILS_H_
#define _COMMON_CDSA_UTILS_H_

#include <Security/cssm.h>
#include <Security/SecKeychain.h>

#ifdef	__cplusplus
extern "C" {
#endif

/* common memory allocators shared by app and CSSM */
extern void * cuAppMalloc (CSSM_SIZE size, void *allocRef);
extern void cuAppFree (void *mem_ptr, void *allocRef);
extern void * cuAppRealloc (void *ptr, CSSM_SIZE size, void *allocRef);
extern void * cuAppCalloc (uint32 num, CSSM_SIZE size, void *allocRef);

#define APP_MALLOC(s)		cuAppMalloc(s, NULL)
#define APP_FREE(p)			cuAppFree(p, NULL)
#define APP_REALLOC(p, s)	cuAppRealloc(p, s, NULL)
#define APP_CALLOC(n, s)	cuAppRealloc(n, s, NULL)

extern CSSM_BOOL cuCompareCssmData(
	const CSSM_DATA *d1,
	const CSSM_DATA *d2);
	
/* OID flavor of same, which will break when an OID is not a CSSM_DATA */
#define cuCompareOid(o1, o2)	cuCompareCssmData(o1, o2)

void cuPrintError(const char *op, CSSM_RETURN err);

/* Init CSSM; returns CSSM_FALSE on error. Reusable. */
extern CSSM_BOOL cuCssmStartup();

/* Attach to CSP. Returns zero on error. */
extern CSSM_CSP_HANDLE cuCspStartup(
	CSSM_BOOL bareCsp);					// true ==> CSP, false ==> CSP/DL

/* Attach to DL side of CSPDL. */
extern CSSM_DL_HANDLE cuDlStartup();

/* Attach to CL, TP */
extern CSSM_CL_HANDLE cuClStartup();
extern CSSM_TP_HANDLE cuTpStartup();

/* Open a DB, ensure it's empty. */
CSSM_DB_HANDLE cuDbStartup(
	CSSM_DL_HANDLE		dlHand,			// from dlStartup()
	const char 			*dbName);

/* Attach to existing DB or create an empty new one. */
CSSM_DB_HANDLE cuDbStartupByName(CSSM_DL_HANDLE dlHand,
	char 		*dbName,
	CSSM_BOOL 	doCreate,
	CSSM_BOOL	quiet);

/* detach and unload */
CSSM_RETURN cuCspDetachUnload(
	CSSM_CSP_HANDLE cspHand,
	CSSM_BOOL bareCsp);					// true ==> CSP, false ==> CSP/DL
CSSM_RETURN cuClDetachUnload(
	CSSM_CL_HANDLE  clHand);
CSSM_RETURN cuDlDetachUnload(
	CSSM_DL_HANDLE  dlHand);
CSSM_RETURN cuTpDetachUnload(
	CSSM_TP_HANDLE  tpHand);
/*
 * Derive symmetric key using PBE.
 */
extern CSSM_RETURN cuCspDeriveKey(CSSM_CSP_HANDLE cspHand,
		uint32				keyAlg,			// CSSM_ALGID_RC5, etc.
		const char 			*keyLabel,
		unsigned 			keyLabelLen,
		uint32 				keyUsage,		// CSSM_KEYUSE_ENCRYPT, etc.
		uint32 				keySizeInBits,
		CSSM_DATA_PTR		password,		// in PKCS-5 lingo
		CSSM_DATA_PTR		salt,			// ditto
		uint32				iterationCnt,	// ditto
		CSSM_KEY_PTR		key);

/*
 * Generate key pair of arbitrary algorithm. 
 */
extern CSSM_RETURN cuCspGenKeyPair(CSSM_CSP_HANDLE cspHand,
	CSSM_DL_DB_HANDLE *dlDbHand,	// optional
	uint32 algorithm,
	const char *keyLabel,
	unsigned keyLabelLen,
	uint32 keySize,					// in bits
	CSSM_KEY_PTR pubKey,			// mallocd by caller
	CSSM_KEYUSE pubKeyUsage,		// CSSM_KEYUSE_ENCRYPT, etc.
	CSSM_KEYATTR_FLAGS pubAttrs,	// CSSM_KEYATTR_EXTRACTABLE, etc. 
	CSSM_KEY_PTR privKey,			// mallocd by caller
	CSSM_KEYUSE privKeyUsage,		// CSSM_KEYUSE_DECRYPT, etc.
	CSSM_KEYATTR_FLAGS privAttrs);	// CSSM_KEYATTR_EXTRACTABLE, etc. 

/* Convert a reference key to a raw key. */
CSSM_RETURN cuRefKeyToRaw(CSSM_CSP_HANDLE cspHand,
	const CSSM_KEY			*refKey,	
	CSSM_KEY_PTR			rawKey);		// RETURNED

/*
 * Add a certificate to a keychain.
 */
CSSM_RETURN cuAddCertToKC(
	SecKeychainRef		keychain,
	const CSSM_DATA		*cert,
	CSSM_CERT_TYPE		certType,
	CSSM_CERT_ENCODING	certEncoding,
	const char			*printName,		// C string
	const CSSM_DATA		*keyLabel);		// ??

/*
 * Convert a CSSM_DATA_PTR, referring to a DER-encoded int, to an
 * unsigned.
 */
unsigned cuDER_ToInt(
	const CSSM_DATA 	*DER_Data);
	
/*
 * Verify a CRL against system anchors and intermediate certs. 
 */
CSSM_RETURN cuCrlVerify(
	CSSM_TP_HANDLE			tpHand, 
	CSSM_CL_HANDLE 			clHand,
	CSSM_CSP_HANDLE 		cspHand,
	const CSSM_DATA			*crlData,
	CSSM_DL_DB_HANDLE_PTR	certKeychain,	// intermediate certs
	const CSSM_DATA 		*anchors,		// optional - if NULL, use Trust Settings
	uint32 					anchorCount);

#ifdef	__cplusplus
}
#endif

#endif	/* _COMMON_CDSA_UTILS_H_ */
