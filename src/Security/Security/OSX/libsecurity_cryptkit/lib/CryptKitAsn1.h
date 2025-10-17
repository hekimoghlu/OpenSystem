/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 25, 2024.
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
#ifndef	_CRYPT_KIT_ASN1_H_
#define _CRYPT_KIT_ASN1_H_

#include "ckconfig.h"


#include <Security/cssmtype.h>
#include <Security/secasn1t.h>

#ifdef	__cplusplus
extern "C" {
#endif

/*
  -- FEE Curve parameters (defined in <security_cryptkit/feeTypes.h>)
	FEEPrimeType ::=    INTEGER { FPT_Mersenne(0), FPT_FEE(1), FPT_General(2) }
	FEECurveType ::=    INTEGER { FCT_Montgomery(0), FCT_Weierstrass(1), 
								  FCT_General(2) }
 */

/*
	FEECurveParameters ::= SEQUENCE
	{
		primeType		FEEPrimeType,
		curveType		FEECurveType,
		q			INTEGER,	-- unsigned
		k			INTEGER,	-- signed 
		m			INTEGER,
		a			BigIntegerStr,
		bb			BigIntegerStr,	-- can't use variable/field b
		c			BigIntegerStr,
		x1Plus			BigIntegerStr,
		x1Minus			BigIntegerStr,
		cOrderPlus		BigIntegerStr,
		cOrderMinus		BigIntegerStr,
		x1OrderPlus		BigIntegerStr,
		x1OrderMinus	BigIntegerStr,
		basePrime		BigIntegerStr OPTIONAL	
										-- iff FEEPrimeType == CT_GENERAL
}
*/
typedef struct {
	CSSM_DATA primeType;
	CSSM_DATA curveType;
	CSSM_DATA q;
	CSSM_DATA k;	
	CSSM_DATA m;	
	CSSM_DATA a;	
	CSSM_DATA b_;			// can't use variable/field b
	CSSM_DATA c;		
	CSSM_DATA x1Plus;		
	CSSM_DATA x1Minus;		
	CSSM_DATA cOrderPlus;	
	CSSM_DATA cOrderMinus;	
	CSSM_DATA x1OrderPlus;	
	CSSM_DATA x1OrderMinus;	
	CSSM_DATA basePrime;		// OPTIONAL	
} FEECurveParametersASN1;

extern const SecAsn1Template FEECurveParametersASN1Template[];

/*
	-- FEE ElGamal-style signature
	FEEElGamalSignature ::= SEQUENCE {
		u     BigIntegerStr,
		pmX 	BigIntegerStr
	}
*/
typedef struct {
	CSSM_DATA	u;
	CSSM_DATA	pmX;
} FEEElGamalSignatureASN1;

extern const SecAsn1Template FEEElGamalSignatureASN1Template[];

/*
	-- FEE ECDSA-style signature
	FEEECDSASignature ::= SEQUENCE {
		c     BigIntegerStr, 
		d     BigIntegerStr
	}
*/
typedef struct {
	CSSM_DATA	c;
	CSSM_DATA	d;
} FEEECDSASignatureASN1;

extern const SecAsn1Template FEEECDSASignatureASN1Template[];

/*
	FEEPublicKey ::= SEQUENCE
	{
		version			INTEGER,
		curveParams		FEECurveParameters,
		plusX			BigIntegerStr,
		minusX			BigIntegerStr,
		plusY			BigIntegerStr	OPTIONAL	
				-- iff FEECurveType == ct-weierstrass
}
*/
typedef struct {
	CSSM_DATA		version;
	FEECurveParametersASN1	curveParams;
	CSSM_DATA		plusX;
	CSSM_DATA		minusX;
	CSSM_DATA		plusY;		// OPTIONAL
} FEEPublicKeyASN1;

extern const SecAsn1Template FEEPublicKeyASN1Template[];

/*
	FEEPrivateKey ::= SEQUENCE 
	{
		version			INTEGER,
		curveParams		FEECurveParameters,
		privData		BigIntegerStr
	}
*/
typedef struct {
	CSSM_DATA		version;
	FEECurveParametersASN1	curveParams;
	CSSM_DATA		privData;
} FEEPrivateKeyASN1;

extern const SecAsn1Template FEEPrivateKeyASN1Template[];

#ifdef	__cplusplus
}
#endif

#endif	/* _CRYPT_KIT_ASN1_H_ */
