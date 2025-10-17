/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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
/* Symbolic constants shared by locate.c and code.c */

#define	NBG		128		/* number of bigrams considered */
#define	OFFSET		14		/* abs value of max likely diff */
#define	PARITY		0200		/* parity bit */
#define	SWITCH		30		/* switch code */
#define UMLAUT          31              /* an 8 bit char followed */

/* 	0-28	likeliest differential counts + offset to make nonnegative */
#define LDC_MIN         0
#define LDC_MAX        28

/*	128-255 bigram codes (128 most common, as determined by 'updatedb') */
#define BIGRAM_MIN    (UCHAR_MAX - SCHAR_MAX) 
#define BIGRAM_MAX    UCHAR_MAX

/*	32-127  single character (printable) ascii residue (ie, literal) */
#define ASCII_MIN      32
#define ASCII_MAX     SCHAR_MAX

/* #define TO7BIT(x)     (x = ( ((u_char)x) & SCHAR_MAX )) */
#define TO7BIT(x)     (x = x & SCHAR_MAX )


#if UCHAR_MAX >= 4096
   define TOLOWER(ch)	  tolower(ch)
#else

extern u_char myctype[UCHAR_MAX + 1];
#define TOLOWER(ch)	(myctype[ch])
#endif

#define INTSIZE (sizeof(int))

#define LOCATE_REG "*?[]\\"  /* fnmatch(3) meta characters */
