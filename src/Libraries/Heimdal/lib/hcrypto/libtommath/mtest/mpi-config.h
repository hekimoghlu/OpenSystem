/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 29, 2025.
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
/* $Id: mpi-config.h,v 1.2 2005/05/05 14:38:47 tom Exp $ */

#ifndef MPI_CONFIG_H_
#define MPI_CONFIG_H_

/*
  For boolean options,
  0 = no
  1 = yes

  Other options are documented individually.

 */

#ifndef MP_IOFUNC
#define MP_IOFUNC     0  /* include mp_print() ?                */
#endif

#ifndef MP_MODARITH
#define MP_MODARITH   1  /* include modular arithmetic ?        */
#endif

#ifndef MP_NUMTH
#define MP_NUMTH      1  /* include number theoretic functions? */
#endif

#ifndef MP_LOGTAB
#define MP_LOGTAB     1  /* use table of logs instead of log()? */
#endif

#ifndef MP_MEMSET
#define MP_MEMSET     1  /* use memset() to zero buffers?       */
#endif

#ifndef MP_MEMCPY
#define MP_MEMCPY     1  /* use memcpy() to copy buffers?       */
#endif

#ifndef MP_CRYPTO
#define MP_CRYPTO     1  /* erase memory on free?               */
#endif

#ifndef MP_ARGCHK
/*
  0 = no parameter checks
  1 = runtime checks, continue execution and return an error to caller
  2 = assertions; dump core on parameter errors
 */
#define MP_ARGCHK     2  /* how to check input arguments        */
#endif

#ifndef MP_DEBUG
#define MP_DEBUG      0  /* print diagnostic output?            */
#endif

#ifndef MP_DEFPREC
#define MP_DEFPREC    64 /* default precision, in digits        */
#endif

#ifndef MP_MACRO
#define MP_MACRO      1  /* use macros for frequent calls?      */
#endif

#ifndef MP_SQUARE
#define MP_SQUARE     1  /* use separate squaring code?         */
#endif

#ifndef MP_PTAB_SIZE
/*
  When building mpprime.c, we build in a table of small prime
  values to use for primality testing.  The more you include,
  the more space they take up.  See primes.c for the possible
  values (currently 16, 32, 64, 128, 256, and 6542)
 */
#define MP_PTAB_SIZE  128  /* how many built-in primes?         */
#endif

#ifndef MP_COMPAT_MACROS
#define MP_COMPAT_MACROS 1   /* define compatibility macros?    */
#endif

#endif /* ifndef MPI_CONFIG_H_ */


/* crc==3287762869, version==2, Sat Feb 02 06:43:53 2002 */

/* $Source: /cvs/libtom/libtommath/mtest/mpi-config.h,v $ */
/* $Revision: 1.2 $ */
/* $Date: 2005/05/05 14:38:47 $ */
