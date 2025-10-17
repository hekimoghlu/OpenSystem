/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 19, 2024.
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
#ifndef	_COMMON_CDSA_UTILS_H_
#define _COMMON_CDSA_UTILS_H_

#include <Security/cssm.h>
#include <Security/SecKeychain.h>
#include <CoreFoundation/CFString.h>

#ifdef	__cplusplus
extern "C" {
#endif

/* common memory allocators shared by app and CSSM */
extern void * srAppMalloc (CSSM_SIZE size, void *allocRef);
extern void srAppFree (void *mem_ptr, void *allocRef);
extern void * srAppRealloc (void *ptr, CSSM_SIZE size, void *allocRef);
extern void * srAppCalloc (uint32 num, CSSM_SIZE size, void *allocRef);

#define APP_MALLOC(s)		srAppMalloc(s, NULL)
#define APP_FREE(p)			srAppFree(p, NULL)
#define APP_REALLOC(p, s)	srAppRealloc(p, s, NULL)
#define APP_CALLOC(n, s)	srAppRealloc(n, s, NULL)

extern CSSM_BOOL srCompareCssmData(
	const CSSM_DATA *d1,
	const CSSM_DATA *d2);
	
/* OID flavor of same, which will break when an OID is not a CSSM_DATA */
#define srCompareOid(o1, o2)	srCompareCssmData(o1, o2)

void srPrintError(const char *op, CSSM_RETURN err);

/* Init CSSM; returns CSSM_FALSE on error. Reusable. */
extern CSSM_BOOL srCssmStartup(void);

/* Attach to CSP. Returns zero on error. */
extern CSSM_CSP_HANDLE srCspStartup(
	CSSM_BOOL bareCsp);					// true ==> CSP, false ==> CSP/DL

/* Attach to DL side of CSPDL. */
extern CSSM_DL_HANDLE srDlStartup(void);

/* Attach to CL, TP */
extern CSSM_CL_HANDLE srClStartup(void);
extern CSSM_TP_HANDLE srTpStartup(void);

/*
 * Derive symmetric key using PBE.
 */
extern CSSM_RETURN srCspDeriveKey(CSSM_CSP_HANDLE cspHand,
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
extern CSSM_RETURN srCspGenKeyPair(CSSM_CSP_HANDLE cspHand,
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
CSSM_RETURN srRefKeyToRaw(CSSM_CSP_HANDLE cspHand,
	const CSSM_KEY			*refKey,	
	CSSM_KEY_PTR			rawKey);		// RETURNED

/*
 * Add a certificate to a keychain.
 */
CSSM_RETURN srAddCertToKC(
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
unsigned srDER_ToInt(
	const CSSM_DATA 	*DER_Data);
	
char *srCfStrToCString(
	CFStringRef cfStr);

#ifdef	__cplusplus
}
#endif

#endif	/* _COMMON_CDSA_UTILS_H_ */
