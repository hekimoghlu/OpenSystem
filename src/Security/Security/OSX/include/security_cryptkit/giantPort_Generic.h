/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 28, 2025.
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
#ifndef	_CK_NSGIANT_PORT_GENERIC_H_
#define _CK_NSGIANT_PORT_GENERIC_H_

#include "feeDebug.h"
#include "platform.h"
#include "giantIntegers.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * We'll be using the compiler's 64-bit long long for these routines.
 *
 * Mask for upper word.
 */
#define GIANT_UPPER_DIGIT_MASK	(~(unsigned long long(GIANT_DIGIT_MASK)))

/*
 * Multiple-precision arithmetic routines/macros.
 */

/*
 * Add two digits, return sum. Carry bit returned as an out parameter.
 * This should work any size giantDigits up to unsigned int.
 */
static inline giantDigit giantAddDigits(
	giantDigit dig1,
	giantDigit dig2,
	giantDigit *carry)			/* RETURNED, 0 or 1 */
{
	giantDigit sum = dig1 + dig2;

	if((sum < dig1) || (sum < dig2)) {
	 	*carry = 1;
	}
	else {
		*carry = 0;
	}
	return sum & GIANT_DIGIT_MASK;
}

/*
 * Add a single digit value to a double digit accumulator in place.
 * Carry out of the MSD of the accumulator is not handled.
 * This should work any size giantDigits up to unsigned int.
 */
static inline void giantAddDouble(
	giantDigit *accLow,			/* IN/OUT */
	giantDigit *accHigh,			/* IN/OUT */
	giantDigit val)
{
	giantDigit sumLo = *accLow + val;

	if((sumLo < *accLow) || (sumLo < val)) {
	    (*accHigh)++;
	    #if	FEE_DEBUG
	    if(*accHigh == 0) {
	        CKRaise("giantAddDouble overflow");
	    }
	    #endif	/* FEE_DEBUG */
	}
	*accLow = sumLo;
}

/*
 * Subtract a - b, return difference. Borrow bit returned as an out parameter.
 * This should work any size giantDigits up to unsigned int.
 */
static inline giantDigit giantSubDigits(
	giantDigit a,
	giantDigit b,
	giantDigit *borrow)			/* RETURNED, 0 or 1 */
{
	giantDigit diff = a - b;

	if(a < b) {
		*borrow = 1;
	}
	else {
		*borrow = 0;
	}
	return diff;
}

/*
 * Multiply two digits, return two digits.
 * This should work for 16 or 32 bit giantDigits, though it's kind of
 * inefficient for 16 bits.
 */
static inline void giantMulDigits(
	giantDigit	dig1,
	giantDigit	dig2,
 	giantDigit	*lowProduct,		/* RETURNED, low digit */
	giantDigit	*hiProduct)		/* RETURNED, high digit */
{
#if GIANT_LOG2_BITS_PER_DIGIT>5
#error "dprod is too small to represent the full result of the multiplication"
#else
	unsigned long long dprod;
#endif

	dprod = (unsigned long long)dig1 * (unsigned long long)dig2;
	*hiProduct =  (giantDigit)(dprod >> GIANT_BITS_PER_DIGIT);
	*lowProduct = (giantDigit)dprod;
}

/*
 * Multiply a vector of giantDigits, candVector, by a single giantDigit,
 * plierDigit, adding results into prodVector. Returns m.s. digit from
 * final multiply; only candLength digits of *prodVector will be written.
 */
static inline giantDigit VectorMultiply(
	giantDigit plierDigit,
	giantDigit *candVector,
	unsigned candLength,
	giantDigit *prodVector)
{
	unsigned candDex;		// index into multiplicandVector
    	giantDigit lastCarry = 0;
	giantDigit prodLo;
	giantDigit prodHi;

	for(candDex=0; candDex<candLength; ++candDex) {
	    /*
	     * prod = *(candVector++) * plierDigit + *prodVector + lastCarry
	     */
	    giantMulDigits(*(candVector++),
		plierDigit,
		&prodLo,
		&prodHi);
	    giantAddDouble(&prodLo, &prodHi, *prodVector);
	    giantAddDouble(&prodLo, &prodHi, lastCarry);

	    /*
	     * *(destptr++) = prodHi;
	     * lastCarry = prodLo;
	     */
	    *(prodVector++) = prodLo;
	    lastCarry = prodHi;
	}

	return lastCarry;
}

#ifdef __cplusplus
extern "C" {
#endif

#endif	/*_CK_NSGIANT_PORT_GENERIC_H_*/
