/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 4, 2024.
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
/* Please send bug reports to David M. Gay (dmg at acm dot org,
 * with " at " changed at "@" and " dot " changed to ".").	*/

#ifndef GDTOA_H_INCLUDED
#define GDTOA_H_INCLUDED

#include "arith.h"
#include <stddef.h> /* for size_t */

#ifndef Long
#define Long long
#endif
#ifndef ULong
typedef unsigned Long ULong;
#endif
#ifndef UShort
typedef unsigned short UShort;
#endif

#ifndef ANSI
#ifdef KR_headers
#define ANSI(x) ()
#define Void /*nothing*/
#else
#define ANSI(x) x
#define Void void
#endif
#endif /* ANSI */

#ifndef CONST
#ifdef KR_headers
#define CONST /* blank */
#else
#define CONST const
#endif
#endif /* CONST */

 enum {	/* return values from strtodg */
	STRTOG_Zero	= 0,
	STRTOG_Normal	= 1,
	STRTOG_Denormal	= 2,
	STRTOG_Infinite	= 3,
	STRTOG_NaN	= 4,
	STRTOG_NaNbits	= 5,
	STRTOG_NoNumber	= 6,
	STRTOG_Retmask	= 7,

	/* The following may be or-ed into one of the above values. */

	STRTOG_Neg	= 0x08, /* does not affect STRTOG_Inexlo or STRTOG_Inexhi */
	STRTOG_Inexlo	= 0x10,	/* returned result rounded toward zero */
	STRTOG_Inexhi	= 0x20, /* returned result rounded away from zero */
	STRTOG_Inexact	= 0x30,
	STRTOG_Underflow= 0x40,
	STRTOG_Overflow	= 0x80
	};

 typedef struct
FPI {
	int nbits;
	int emin;
	int emax;
	int rounding;
	int sudden_underflow;
	} FPI;

enum {	/* FPI.rounding values: same as FLT_ROUNDS */
	FPI_Round_zero = 0,
	FPI_Round_near = 1,
	FPI_Round_up = 2,
	FPI_Round_down = 3
	};

#ifdef __cplusplus
extern "C" {
#endif

extern char* dtoa  ANSI((double d, int mode, int ndigits, int *decpt,
			int *sign, char **rve));
extern char* gdtoa ANSI((FPI *fpi, int be, ULong *bits, int *kindp,
			int mode, int ndigits, int *decpt, char **rve));
extern void freedtoa ANSI((char*));
extern float  strtof ANSI((CONST char *, char **));
extern double strtod ANSI((CONST char *, char **));
extern int strtodg ANSI((CONST char*, char**, CONST FPI*, Long*, ULong*, locale_t)) __DARWIN_ALIAS(strtodg);

extern char*	g_ddfmt  ANSI((char*, double*, int, size_t));
extern char*	g_dfmt   ANSI((char*, double*, int, size_t));
extern char*	g_ffmt   ANSI((char*, float*,  int, size_t));
extern char*	g_Qfmt   ANSI((char*, void*,   int, size_t));
extern char*	g_xfmt   ANSI((char*, void*,   int, size_t));
extern char*	g_xLfmt  ANSI((char*, void*,   int, size_t));

extern int	strtoId  ANSI((CONST char*, char**, double*, double*));
extern int	strtoIdd ANSI((CONST char*, char**, double*, double*));
extern int	strtoIf  ANSI((CONST char*, char**, float*, float*));
extern int	strtoIQ  ANSI((CONST char*, char**, void*, void*));
extern int	strtoIx  ANSI((CONST char*, char**, void*, void*));
extern int	strtoIxL ANSI((CONST char*, char**, void*, void*));
extern int	strtord  ANSI((CONST char*, char**, int, double*));
extern int	strtordd ANSI((CONST char*, char**, int, double*));
extern int	strtorf  ANSI((CONST char*, char**, int, float*));
extern int	strtorQ  ANSI((CONST char*, char**, int, void*));
extern int	strtorx  ANSI((CONST char*, char**, int, void*));
extern int	strtorxL ANSI((CONST char*, char**, int, void*));
#if 1
extern int	strtodI  ANSI((CONST char*, char**, double*));
extern int	strtopd  ANSI((CONST char*, char**, double*));
extern int	strtopdd ANSI((CONST char*, char**, double*, locale_t));
extern int	strtopf  ANSI((CONST char*, char**, float*));
extern int	strtopQ  ANSI((CONST char*, char**, void*));
extern int	strtopx  ANSI((CONST char*, char**, void*, locale_t));
extern int	strtopxL ANSI((CONST char*, char**, void*));
#else
#define strtopd(s,se,x) strtord(s,se,1,x)
#define strtopdd(s,se,x) strtordd(s,se,1,x)
#define strtopf(s,se,x) strtorf(s,se,1,x)
#define strtopQ(s,se,x) strtorQ(s,se,1,x)
#define strtopx(s,se,x) strtorx(s,se,1,x)
#define strtopxL(s,se,x) strtorxL(s,se,1,x)
#endif

#ifdef __cplusplus
}
#endif
#endif /* GDTOA_H_INCLUDED */
