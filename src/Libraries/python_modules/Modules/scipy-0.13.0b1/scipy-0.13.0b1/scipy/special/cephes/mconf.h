/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 16, 2022.
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
 * Cephes Math Library Release 2.3:  June, 1995
 * Copyright 1984, 1987, 1989, 1995 by Stephen L. Moshier
 */

#ifndef CEPHES_MCONF_H
#define CEPHES_MCONF_H

#include <Python.h>
#include <numpy/npy_math.h>

#include "cephes_names.h"
#include "protos.h"

/* Constant definitions for math error conditions
 */

#define DOMAIN		1	/* argument domain error */
#define SING		2	/* argument singularity */
#define OVERFLOW	3	/* overflow range error */
#define UNDERFLOW	4	/* underflow range error */
#define TLOSS		5	/* total loss of precision */
#define PLOSS		6	/* partial loss of precision */
#define TOOMANY         7	/* too many iterations */
#define MAXITER        500

#define EDOM		33
#define ERANGE		34

/* Long double complex numeral.  */
/*
 * typedef struct
 * {
 * long double r;
 * long double i;
 * } cmplxl;
 */

/* Type of computer arithmetic */

/* UNKnown arithmetic, invokes coefficients given in
 * normal decimal format.  Beware of range boundary
 * problems (MACHEP, MAXLOG, etc. in const.c) and
 * roundoff problems in pow.c:
 * (Sun SPARCstation)
 */

/* SciPy note: by defining UNK, we prevent the compiler from
 * casting integers to floating point numbers.  If the Endianness
 * is detected incorrectly, this causes problems on some platforms.
 */
#define UNK 1

/* Define this `volatile' if your compiler thinks
 * that floating point arithmetic obeys the associative
 * and distributive laws.  It will defeat some optimizations
 * (but probably not enough of them).
 *
 * #define VOLATILE volatile
 */
#define VOLATILE

/* For 12-byte long doubles on an i386, pad a 16-bit short 0
 * to the end of real constants initialized by integer arrays.
 *
 * #define XPD 0,
 *
 * Otherwise, the type is 10 bytes long and XPD should be
 * defined blank (e.g., Microsoft C).
 *
 * #define XPD
 */
#define XPD 0,

/* Define to support tiny denormal numbers, else undefine. */
#define DENORMAL 1

/* Define to distinguish between -0.0 and +0.0.  */
#define MINUSZERO 1

/* Define 1 for ANSI C atan2() function
 * See atan.c and clog.c. */
#define ANSIC 1

/* Variable for error reporting.  See mtherr.c.  */
extern int merror;

#define gamma Gamma

#endif				/* CEPHES_MCONF_H */
