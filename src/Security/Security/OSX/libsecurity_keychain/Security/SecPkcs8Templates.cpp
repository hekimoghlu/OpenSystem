/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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
 * SecPkcs8Templates.cpp - ASN1 templates for private keys in PKCS8 format.  
 */

#include "SecPkcs8Templates.h"
#include <Security/keyTemplates.h>
#include <Security/secasn1t.h>
#include <security_asn1/prtypes.h>  
#include <stddef.h>

const SecAsn1Template impExpPKCS5_PBE_ParametersTemplate[] = {
	{ SEC_ASN1_SEQUENCE,
      0, NULL, sizeof(impExpPKCS5_PBE_Parameters) },
    { SEC_ASN1_OCTET_STRING,
	  offsetof(impExpPKCS5_PBE_Parameters,salt) },
	/* iterations is unsigned - right? */
	{ SEC_ASN1_INTEGER,
	  offsetof(impExpPKCS5_PBE_Parameters,iterations) },
	{ 0 }
};

const SecAsn1Template impExpPKCS5_PBKDF2_ParamsTemplate[] = {
	{ SEC_ASN1_SEQUENCE,
      0, NULL, sizeof(impExpPKCS5_PBKDF2_Params) },
    { SEC_ASN1_OCTET_STRING,
	  offsetof(impExpPKCS5_PBKDF2_Params,salt) },
	/* iterations is unsigned - right? */
	{ SEC_ASN1_INTEGER,
	  offsetof(impExpPKCS5_PBKDF2_Params,iterationCount) },
	{ SEC_ASN1_INTEGER | SEC_ASN1_OPTIONAL,
	  offsetof(impExpPKCS5_PBKDF2_Params,keyLengthInBytes) },
	{ SEC_ASN1_OBJECT_ID | SEC_ASN1_OPTIONAL,
	  offsetof(impExpPKCS5_PBKDF2_Params,prf) },
	{ 0 }
};

const SecAsn1Template impExpPKCS5_RC2ParamsTemplate[] = {
	{ SEC_ASN1_SEQUENCE,
      0, NULL, sizeof(impExpPKCS5_RC2Params) },
	{ SEC_ASN1_INTEGER | SEC_ASN1_OPTIONAL,
	  offsetof(impExpPKCS5_RC2Params,version) },
    { SEC_ASN1_OCTET_STRING,
	  offsetof(impExpPKCS5_RC2Params,iv) },
	{ 0 }
};

const SecAsn1Template impExpPKCS5_RC5ParamsTemplate[] = {
	{ SEC_ASN1_SEQUENCE,
      0, NULL, sizeof(impExpPKCS5_RC5Params) },
	{ SEC_ASN1_INTEGER,
	  offsetof(impExpPKCS5_RC5Params,version) },
	{ SEC_ASN1_INTEGER,
	  offsetof(impExpPKCS5_RC5Params,rounds) },
	{ SEC_ASN1_INTEGER,
	  offsetof(impExpPKCS5_RC5Params,blockSizeInBits) },
    { SEC_ASN1_OCTET_STRING,
	  offsetof(impExpPKCS5_RC5Params,iv) },
	{ 0 }
};

const SecAsn1Template impExpPKCS5_PBES2_ParamsTemplate[] = {
	{ SEC_ASN1_SEQUENCE,
      0, NULL, sizeof(impExpPKCS5_PBES2_Params) },
    { SEC_ASN1_INLINE,
	  offsetof(impExpPKCS5_PBES2_Params,keyDerivationFunc),
	  kSecAsn1AlgorithmIDTemplate },
    { SEC_ASN1_INLINE,
	  offsetof(impExpPKCS5_PBES2_Params,encryptionScheme),
	  kSecAsn1AlgorithmIDTemplate },
	{ 0 }
};
