/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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
 * SecPkcs8Templates.h - ASN1 templates for private keys in PKCS8 format.  
 */
 
#ifndef _SEC_PKCS8_TEMPLATES_H_
#define _SEC_PKCS8_TEMPLATES_H_

#include <Security/cssmtype.h>
#include <Security/x509defs.h>
#include <Security/secasn1t.h>

#ifdef  __cplusplus
extern "C" {
#endif

/*
 * This one is the AlgorithmID.Parameters field for PKCS5 v1.5.
 * It looks mighty similar to pkcs-12PbeParams except that this 
 * one has a fixed salt size of 8 bytes (not that we enforce that
 * at decode time). 
 */
typedef struct {
	CSSM_DATA		salt;
	CSSM_DATA		iterations;
} impExpPKCS5_PBE_Parameters;

extern const SecAsn1Template impExpPKCS5_PBE_ParametersTemplate[];

/*
 * This is the AlgorithmID.Parameters of the keyDerivationFunc component
 * of a PBES2-params. PKCS v2.0 only. We do not handle the CHOICE salt;
 * only the specified flavor (as an OCTET STRING).
 */
typedef struct {
	CSSM_DATA		salt;
	CSSM_DATA		iterationCount;
	CSSM_DATA		keyLengthInBytes;	// optional
	CSSM_OID		prf;				// optional, default algid-hmacWithSHA1
} impExpPKCS5_PBKDF2_Params;

extern const SecAsn1Template impExpPKCS5_PBKDF2_ParamsTemplate[];

/*
 * AlgorithmID.Parameters for encryptionScheme component of of a PBES2-params.
 * This one for RC2:
 */
typedef struct {
	CSSM_DATA		version;		// optional
	CSSM_DATA		iv;				// 8 bytes
} impExpPKCS5_RC2Params;

extern const SecAsn1Template impExpPKCS5_RC2ParamsTemplate[];

/*
 * This one for RC5.
 */
typedef struct {
	CSSM_DATA		version;			// not optional
	CSSM_DATA		rounds;				// 8..127
	CSSM_DATA		blockSizeInBits;	// 64 | 128
	CSSM_DATA		iv;					// optional, default is all zeroes
} impExpPKCS5_RC5Params;

extern const SecAsn1Template impExpPKCS5_RC5ParamsTemplate[];

/*
 * The top-level AlgID.Parameters for PKCS5 v2.0. 
 * keyDerivationFunc.Parameters is a impExpPKCS5_PBKDF2_Params.
 * encryptionScheme.Parameters depends on the encryption algorithm:
 *
 * DES, 3DES: encryptionScheme.Parameters is an OCTET STRING containing the 
 *            8-byte IV. 
 * RC2: encryptionScheme.Parameters is impExpPKCS5_RC2Params.
 * RC5: encryptionScheme.Parameters is impExpPKCS5_RC5Params.
 */
typedef struct {
	CSSM_X509_ALGORITHM_IDENTIFIER  keyDerivationFunc;
	CSSM_X509_ALGORITHM_IDENTIFIER  encryptionScheme;
} impExpPKCS5_PBES2_Params;

extern const SecAsn1Template impExpPKCS5_PBES2_ParamsTemplate[];

#ifdef  __cplusplus
}
#endif

#endif  /* _SEC_PKCS8_TEMPLATES_H_ */
