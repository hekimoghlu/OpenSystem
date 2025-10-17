/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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
 * @OSF_COPYRIGHT@
 */
/*
 * HISTORY
 *
 * Revision 1.1.1.1  1998/09/22 21:05:51  wsanchez
 * Import of Mac OS X kernel (~semeria)
 *
 * Revision 1.1.1.1  1998/03/07 02:25:35  wsanchez
 * Import of OSF Mach kernel (~mburg)
 *
 * Revision 1.1.2.1  1996/09/17  16:56:21  bruel
 *      created from standalone mach servers.
 *      [96/09/17            bruel]
 *
 * $EndLog$
 */

#ifndef _MACH_TYPES_H_
#define _MACH_TYPES_H_

#include <stddef.h>
#include "libsa/machine/types.h"

#ifndef _SIZE_T
#define _SIZE_T
typedef unsigned long   size_t;
#endif  /* _SIZE_T */

/*
 * Common type definitions that lots of old files seem to want.
 */

typedef unsigned char   u_char;         /* unsigned char */
typedef unsigned short  u_short;        /* unsigned short */
typedef unsigned int    u_int;          /* unsigned int */
typedef unsigned long   u_long;         /* unsigned long */

typedef struct _quad_ {
	unsigned int    val[2];         /* 2 32-bit values make... */
} quad;                                 /* an 8-byte item */

typedef char *          caddr_t;        /* address of a (signed) char */

typedef unsigned int    daddr_t;        /* an unsigned 32 */
#if 0 /* off_t should be 64-bit ! */
typedef unsigned int    off_t;          /* another unsigned 32 */
#endif


#define major(i)        (((i) >> 8) & 0xFF)
#define minor(i)        ((i) & 0xFF)
#define makedev(i, j)    ((((i) & 0xFF) << 8) | ((j) & 0xFF))

#ifndef NULL
#define NULL            ((void *) 0)    /* the null pointer */
#endif

/*
 * Shorthand type definitions for unsigned storage classes
 */
typedef unsigned char   uchar_t;
typedef unsigned short  ushort_t;
typedef unsigned int    uint_t;
typedef unsigned long   ulong_t;
typedef volatile unsigned char  vuchar_t;
typedef volatile unsigned short vushort_t;
typedef volatile unsigned int   vuint_t;
typedef volatile unsigned long  vulong_t;

/*
 * Deprecation macro
 */
#if __GNUC__ >= 3
#define __deprecated    __attribute__((__deprecated__))
#else
#define __deprecated /* nothing */
#endif

#endif  /* _MACH_TYPES_H_ */
