/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 22, 2023.
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
#define Long int
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
	STRTOG_Zero	= 0x000,
	STRTOG_Normal	= 0x001,
	STRTOG_Denormal	= 0x002,
	STRTOG_Infinite	= 0x003,
	STRTOG_NaN	= 0x004,
	STRTOG_NaNbits	= 0x005,
	STRTOG_NoNumber	= 0x006,
	STRTOG_NoMemory = 0x007,
	STRTOG_Retmask	= 0x00f,

	/* The following may be or-ed into one of the above values. */

	STRTOG_Inexlo	= 0x010, /* returned result rounded toward zero */
	STRTOG_Inexhi	= 0x020, /* returned result rounded away from zero */
	STRTOG_Inexact	= 0x030,
	STRTOG_Underflow= 0x040,
	STRTOG_Overflow	= 0x080,
	STRTOG_Neg	= 0x100 /* does not affect STRTOG_Inexlo or STRTOG_Inexhi */
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

extern char* __dtoa  ANSI((double d, int mode, int ndigits, int *decpt,
			int *sign, char **rve));
extern char* __gdtoa ANSI((FPI *fpi, int be, ULong *bits, int *kindp,
			int mode, int ndigits, int *decpt, char **rve));
extern void __freedtoa ANSI((char*));
extern float  strtof ANSI((CONST char *, char **));
extern double strtod ANSI((CONST char *, char **));
extern int __strtodg ANSI((CONST char*, char**, FPI*, Long*, ULong*));
char	*__hdtoa(double, const char *, int, int *, int *, char **);
char	*__hldtoa(long double, const char *, int, int *, int *, char **);
char	*__ldtoa(long double *, int, int, int *, int *, char **);

PROTO_NORMAL(__dtoa);
PROTO_NORMAL(__gdtoa);
PROTO_NORMAL(__freedtoa);
PROTO_NORMAL(__hdtoa);
PROTO_NORMAL(__hldtoa);
PROTO_NORMAL(__ldtoa);

__BEGIN_HIDDEN_DECLS
extern char*	__g_ddfmt  ANSI((char*, double*, int, size_t));
extern char*	__g_dfmt   ANSI((char*, double*, int, size_t));
extern char*	__g_ffmt   ANSI((char*, float*,  int, size_t));
extern char*	__g_Qfmt   ANSI((char*, void*,   int, size_t));
extern char*	__g_xfmt   ANSI((char*, void*,   int, size_t));
extern char*	__g_xLfmt  ANSI((char*, void*,   int, size_t));

extern int	__strtoId  ANSI((CONST char*, char**, double*, double*));
extern int	__strtoIdd ANSI((CONST char*, char**, double*, double*));
extern int	__strtoIf  ANSI((CONST char*, char**, float*, float*));
extern int	__strtoIQ  ANSI((CONST char*, char**, void*, void*));
extern int	__strtoIx  ANSI((CONST char*, char**, void*, void*));
extern int	__strtoIxL ANSI((CONST char*, char**, void*, void*));
extern int	__strtord  ANSI((CONST char*, char**, int, double*));
extern int	__strtordd ANSI((CONST char*, char**, int, double*));
extern int	__strtorf  ANSI((CONST char*, char**, int, float*));
extern int	__strtorQ  ANSI((CONST char*, char**, int, void*));
extern int	__strtorx  ANSI((CONST char*, char**, int, void*));
extern int	__strtorxL ANSI((CONST char*, char**, int, void*));
#if 1
extern int	__strtodI  ANSI((CONST char*, char**, double*));
extern int	__strtopd  ANSI((CONST char*, char**, double*));
extern int	__strtopdd ANSI((CONST char*, char**, double*));
extern int	__strtopf  ANSI((CONST char*, char**, float*));
extern int	__strtopQ  ANSI((CONST char*, char**, void*));
extern int	__strtopx  ANSI((CONST char*, char**, void*));
extern int	__strtopxL ANSI((CONST char*, char**, void*));
#else
#define __strtopd(s,se,x) strtord(s,se,1,x)
#define __strtopdd(s,se,x) strtordd(s,se,1,x)
#define __strtopf(s,se,x) strtorf(s,se,1,x)
#define __strtopQ(s,se,x) strtorQ(s,se,1,x)
#define __strtopx(s,se,x) strtorx(s,se,1,x)
#define __strtopxL(s,se,x) strtorxL(s,se,1,x)
#endif
__END_HIDDEN_DECLS

#ifdef __cplusplus
}
#endif
#endif /* GDTOA_H_INCLUDED */
