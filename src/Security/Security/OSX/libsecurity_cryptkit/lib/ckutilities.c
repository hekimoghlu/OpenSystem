/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 10, 2023.
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
#include "ckutilities.h"
#include "falloc.h"
#include "feeTypes.h"
#include "feeDebug.h"
#include "feeFunctions.h"
#include "byteRep.h"
#include "platform.h"
#include "curveParams.h"
#include <stdlib.h>
#ifdef	NeXT
#include <libc.h>
#include <stdio.h>
#include <signal.h>
#include <sgtty.h>
#endif	// NeXT

/*
 * feeReturn strings.
 */
typedef struct {
	feeReturn	frtn;
	const char	*frtnString;
} frtnItem;

static const frtnItem frtnStrings[] = {
#ifndef	NDEBUG
	{ FR_Success,				"Success"					},
	{ FR_BadPubKey,				"Bad Public Key"			},
	{ FR_BadPubKeyString,		"Bad Public Key String"		},
	{ FR_IncompatibleKey,		"Incompatible key format"	},
	{ FR_IllegalDepth,			"Illegal Depth"				},
	{ FR_BadUsageName,			"Bad Usage Name"			},
	{ FR_BadSignatureFormat, 	"Bad Signature Format"		},
	{ FR_InvalidSignature,		"Invalid Signature"			},
	{ FR_IllegalArg,			"Illegal Argument" 			},
	{ FR_BadCipherText,			"Bad Ciphertext Format"		},
	{ FR_Unimplemented,			"Unimplemented Function"	},
	{ FR_BadCipherFile,			"Bad CipherFile Format"		},
	{ FR_BadEnc64,				"Bad enc64 Format"			},
	{ FR_WrongSignatureType, 	"Wrong Signature Type"		},
	{ FR_BadKeyBlob,			"Bad Key Blob"				},
	{ FR_IllegalCurve,			"Bad curve type"			},
	{ FR_Internal,				"Internal Library Error"	},
	{ FR_Memory, 				"Out of Memory"				},
	{ FR_ShortPrivData,			"Insufficient Seed Data" 	},
#endif	/* NDEBUG */
	{ (feeReturn) 0,			NULL						},
};

/*
 * One-time only init of CryptKit library.
 */
void initCryptKit(void)
{
}

/*
 * Shutdown.
 */
void terminateCryptKit(void)
{
}

/*
 * Create a giant, initialized with specified char[] data.
 */
giant giant_with_data(const unsigned char *d, int len) {
    int numDigits = BYTES_TO_GIANT_DIGITS(len);
    giant result;

    result = newGiant(numDigits);
    deserializeGiant(d, result, len);
    return result;
}

/*
 * Obtain a malloc'd memory chunk init'd with specified giant's data.
 * Resulting bytes are portable. Size of malloc'd memory is always zero
 * mod GIANT_BYTES_PER_DIGIT.
 *
 * Calling this function for a giant obtained by giant_with_data() yields
 * the original data, with extra byte(s) of leading zeros if the original
 * was not zero mod GIANT_BYTES_PER_DIGIT.
 */
unsigned char *mem_from_giant(giant g,
	unsigned *memLen)		/* RETURNED size of malloc'd region */
{
	unsigned char *cp;
	unsigned numDigits = (g->sign < 0) ? -g->sign : g->sign;

	*memLen = numDigits * GIANT_BYTES_PER_DIGIT;
	cp = (unsigned char*) fmalloc(*memLen);
	serializeGiant(g, cp, *memLen);
	return cp;
}

extern const char *feeReturnString(feeReturn frtn)
{
	const frtnItem *fi = frtnStrings;

	while(fi->frtnString) {
		if(fi->frtn == frtn) {
			return fi->frtnString;
		}
		fi++;
	}
	return "Unknown Status";
}

#if		FEE_DEBUG
void printGiant(const giant x)
{
	int i;

	printf("sign=%d cap=%d n[]=", x->sign, x->capacity);
	for(i=0; i<abs(x->sign); i++) {
		printf("%lu:", (unsigned long)x->n[i]);
	}
	printf("\n");
}

void printGiantHex(const giant x)
{
	int i;

	printf("sign=%d cap=%d n[]=", x->sign, x->capacity);
	for(i=0; i<abs(x->sign); i++) {
		printf("%lx:", (unsigned long)x->n[i]);
	}
	printf("\n");
}

/*
 * Print in the form
 *   sign=8 cap=16 n[]=29787 + 3452 * w^1 + 55260 * w^2  + ...
 */
void printGiantExp(const giant x)
{
	int i;
	int size = abs(x->sign);

	printf("sign=%d cap=%d n[]=", x->sign, x->capacity);
	for(i=0; i<size; i++) {
		printf("%lu ", (unsigned long)x->n[i]);
		if(i > 0) {
			printf("* w^%d ", i);
		}
		if(i<(size-1)) {
			printf("+ ");
		}
	}
	printf("\n");
}

void printKey(const key k)
{
	printf("  twist %d\n", k->twist);
	printf("  x: ");
	printGiant(k->x);
}

void printCurveParams(const curveParams *p)
{
	const char *pt;
	const char *ct;
	
	switch(p->primeType) {
	    case FPT_Mersenne:
	    	pt = "FPT_Mersenne";
		break;
	    case FPT_FEE:
	    	pt = "FPT_FEE";
		break;
	    case FPT_General:
	    	pt = "FPT_General";
		break;
	    default:
	    	pt = "UNKNOWN!";
		break;
	}
	switch(p->curveType) {
		case FCT_Montgomery:
			ct = "FCT_Montgomery";
			break;
		case FCT_Weierstrass:
			ct = "FCT_Weierstrass";
			break;
		case FCT_General:
			ct = "FCT_General";
			break;
	    default:
	    	ct = "UNKNOWN!";
			break;
	}
	printf("  q %d   k %d   primeType %s  curveType %s\n",
		p->q, p->k, pt, ct);
	printf("  minBytes %d  maxDigits %d\n", p->minBytes, p->maxDigits);
	printf("  a           : ");
	printGiant(p->a);
	printf("  b           : ");
	printGiant(p->b);
	printf("  c           : ");
	printGiant(p->c);
	printf("  basePrime   : ");
	printGiant(p->basePrime);
	printf("  x1Plus      : ");
	printGiant(p->x1Plus);
	printf("  x1Minus     : ");
	printGiant(p->x1Minus);
	printf("  cOrderPlus  : ");
	printGiant(p->cOrderPlus);
	printf("  cOrderMinus : ");
	printGiant(p->cOrderMinus);
	printf("  x1OrderPlus : ");
	printGiant(p->x1OrderPlus);
	printf("  x1OrderMinus: ");
	printGiant(p->x1OrderMinus);
}
#else
void printGiant(const giant x) {}
void printGiantHex(const giant x) {}
void printGiantExp(const giant x) {}
void printKey(const key k) {}
void printCurveParams(const curveParams *p) {}

#endif	/* FEE_DEBUG */

/*
 * serialize, deserialize giants's n[] to/from byte stream.
 * First byte of byte stream is the MS byte of the resulting giant,
 * regardless of the size of giantDigit.
 *
 * No assumption is made about the alignment of cp.
 *
 * As of 7 Apr 1998, these routines are in compliance with IEEE P1363,
 * section 5.5.1, for the representation of a large integer as a byte
 * stream.
 */
void serializeGiant(giant g,
	unsigned char *cp,
	unsigned numBytes)
{
	unsigned	digitDex;
	unsigned 	numDigits = BYTES_TO_GIANT_DIGITS(numBytes);
	giantDigit 	digit;
	unsigned char 	*ptr;
	unsigned	digitByte;
	int 		size = abs(g->sign);

	if(numBytes == 0) {
		return;
	}
	if(numBytes > (g->capacity * GIANT_BYTES_PER_DIGIT)) {
		CKRaise("serializeGiant: CAPACITY EXCEEDED!\n");
	}

	/*
	 * note we might be asked to write more than the valid number
	 * if bytes in the giant in the case if truncated sign due to
	 * zero M.S. digit(s)....
	 */

	/*
	 * zero out unused digits so we can infer sign during deserialize
	 */
	for(digitDex=size; digitDex<numDigits; digitDex++) {
		g->n[digitDex] = 0;
	}

	/*
	 * Emit bytes starting from l.s. byte. L.s. byte of the outgoing
	 * data stream is *last*. L.s. digit of giant's digits is *first*.
	 */
	digitDex = 0;
	ptr = &cp[numBytes - 1];
	do {
	    /* one loop per giant digit */
	    digit = g->n[digitDex++];
	    for(digitByte=0; digitByte<GIANT_BYTES_PER_DIGIT; digitByte++) {
	        /* one loop per byte in the digit */
	    	*ptr-- = (unsigned char)digit;
			if(--numBytes == 0) {
				break;
			}
			digit >>= 8;
	    }
	} while(numBytes != 0);

}

/*
 * Resulting sign here is always positive; leading zeroes are reflected
 * in an altered g->sign.
 */
void deserializeGiant(const unsigned char *cp,
	giant g,
	unsigned numBytes)
{
	unsigned 		numDigits;
	giantDigit 		digit;
	int				digitDex;
	unsigned		digitByte;
	const unsigned char 	*ptr;

	if(numBytes == 0) {
		g->sign = 0;
		return;
	}
	numDigits = (numBytes + GIANT_BYTES_PER_DIGIT - 1) /
			GIANT_BYTES_PER_DIGIT;
	if(numBytes > (g->capacity * GIANT_BYTES_PER_DIGIT)) {
		CKRaise("deserializeGiant: CAPACITY EXCEEDED!\n");
	}

	/*
	 * Start at l.s. byte. That's the end of the cp[] array and
	 * the beginning of the giantDigit array.
	 */
	digitDex = 0;
	ptr = &cp[numBytes - 1];
	do {
	    /* one loop per digit */
	    digit = 0;
	    for(digitByte=0; digitByte<GIANT_BYTES_PER_DIGIT; digitByte++) {
	        /* one loop per byte in the digit */
		digit |= (*ptr-- << (8 * digitByte));
		/* FIXME - shouldn't we update g->n before this break? */
		if(--numBytes == 0) {
		    break;
		}
	    }
	    g->n[digitDex++] = digit;
	} while (numBytes != 0);

	/*
	 * Infer sign from non-zero n[] elements
	 */
	g->sign = numDigits;
	gtrimSign(g);
}

