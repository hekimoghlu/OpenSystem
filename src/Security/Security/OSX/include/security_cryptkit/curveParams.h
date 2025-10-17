/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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
#ifndef	_CK_CURVEPARAMS_H_
#define _CK_CURVEPARAMS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "giantIntegers.h"
#include "feeTypes.h"

/*
 * Parameters defining a specific elliptic curve (and its initial points).
 */
typedef struct {

	/*
	 * Basic characteristic of prime field (PT_FEE, etc.)
	 */
	feePrimeType	primeType;

	/*
	 * Basic curve type (CT_MONTGOMERY, etc.)
	 * Note that FCT_ANSI is stored here as FCT_Weierstrass.
	 */
	feeCurveType	curveType;

	/*
	 * Parameters defining the base prime (2^q - k) for
	 * FPT_FEE and FPT_Mersenne. For FPT_General, q is the 
	 * prime size in bits and k is 0.
	 */
	unsigned	q;
	int         k;

	/*
	 * For all primeTypes, the field is defined as F(basePrime**m).
	 * This library can only deal with m == 1 for now.
	 */
	unsigned	m;

	/*
	 * coefficients in the following equation:
	 * y^2 = x^3 + (c * x^2) + (a * x) + b
	 */
	giant 		a;
	giant 		b;
	giant 		c;

	/*
	 * Initial public point x-coordinates.
	 * x1Minus not used for ECDSA; X9.62 curves don't have this field. 
	 */
	giant		x1Plus;
	giant		x1Minus;

	/*
	 * Y coordinate of normalized projective initial public
	 * point for plus curve. I.e., Initial point = {x1Plus, p1Plus, 1}.
	 * Only valid for curveType == CT_WEIERSTRASS. This is calculated
	 * when a new curveParams is created.
	 */
	giant		y1Plus;

	/*
	 * Curve orders. These are prime, or have large prime factors.
	 * cOrderMinus not used for ECDSA; X9.62 curves don't have this field. 
	 */
	giant		cOrderPlus;
	giant		cOrderMinus;

	/*
	 * Point orders (the large prime factors of the respective
	 * curve orders).
	 * x1OrderMinus not used for ECDSA; X9.62 curves don't have this field. 
	 */
	giant		x1OrderPlus;
	giant		x1OrderMinus;

	/*
	 * The base prime. For PT_GENERAL, this is a basic defining
	 * characteristic of a curve; otherwise, it is derived as 2**q - k.
	 */
	giant		basePrime;

	/*
	 * The remaining fields are calculated and stored here as an
	 * optimization.
	 */

	/*
	 * The minimum size of a giant, in bytes, to represent any point
	 * on this curve. This is generally used only when serializing
	 * giants of a known size.
	 */
	unsigned	minBytes;

	/*
	 * The maximum size of a giant, in giantDigits, to be used with all
	 * FEE arithmetic for this curve. This is generally used to alloc
	 * giants.
	 */
	unsigned	maxDigits;

	/*
	 * Reciprocals of lesserX1Order() and x1OrderPlus. Calculated
	 * lazily by clients in the case of creation of a curveParams
	 * struct from a byteRep representation.
	 */
	giant		x1OrderPlusRecip;
	giant		lesserX1OrderRecip;

	/*
	 * Reciprocal of basePrime. Only used for PT_GENERAL.
	 */
	giant		basePrimeRecip;
} curveParams;

#if 0
/*
 * Values for primeType.
 */
#define PT_MERSENNE	0	/* basePrime = 2**q - 1 */
#define PT_FEE		1	/* basePrime = 2**q - k, k is "small" */
#define PT_GENERAL	2	/* other prime modulus */

/*
 * Values for curveType. Note that Atkin3 (a=0) and Atkin4 (b=0) are
 * subsets of CT_WEIERSTRASS.
 */
#define CT_MONTGOMERY	0	/* a=1, b=0 */
#define CT_WEIERSTRASS	1	/* c=0 */
#define CT_GENERAL		4	/* other */
#endif /* 0 */

/*
 * Obtain a malloc'd curveParams for a specified feeDepth.
 */
curveParams *curveParamsForDepth(feeDepth depth);

/*
 * Obtain a malloc'd and uninitialized curveParams, to be init'd by caller
 * (when matching existing curve params).
 */
curveParams *newCurveParams(void);

/*
 * Alloc and zero reciprocal giants, when maxDigits is known.
 */
void allocRecipGiants(curveParams *cp);

/*
 * Alloc a new curveParams struct as a copy of specified instance.
 */
curveParams *curveParamsCopy(curveParams *cp);

/*
 * Free a curveParams struct.
 */
void freeCurveParams(curveParams *cp);

/*
 * Returns 1 if two sets of curve parameters are equivalent, else returns 0.
 */
int curveParamsEquivalent(curveParams *cp1, curveParams *cp2);

/*
 * Obtain the lesser of {x1OrderPlus, x1OrderMinus}. Returned value is not
 * malloc'd; it's a pointer to one of the orders in *cp.
 */
giant lesserX1Order(curveParams *cp);

/*
 * Prime the curveParams and giants modules for quick allocs of giants.
 */
void curveParamsInitGiants(void);

/*
 * Infer run-time calculated fields from a partially constructed curveParams.
 */
void curveParamsInferFields(curveParams *cp);

/*
 * Given key size in bits, obtain the asssociated depth.
 * Returns FR_IllegalDepth if specify key size not found
 * in current curve tables. 
 */
feeReturn feeKeyBitsToDepth(unsigned keyBits,
	feePrimeType primeType,		/* FPT_Fefault means "best one" */
	feeCurveType curveType,		/* FCT_Default means "best one" */
	feeDepth *depth);
	
/* 
 * Obtain depth for specified curveParams
 */
feeReturn curveParamsDepth(
	curveParams *cp,
	feeDepth *depth);
	
#ifdef __cplusplus
}
#endif

#endif	/* _CK_CURVEPARAMS_H_ */
