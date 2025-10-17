/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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
**  NAME:
**
**      ndrtypes.h
**
**  FACILITY:
**
**      IDL Stub Support Include File
**
**  ABSTRACT:
**
**  This file is new for DCE 1.1. This is a platform specific file that
**  defines the base level ndr types. This file is indirectly included
**  in all files via the idlbase.h file.
**
*/

/*
 * This particular file defines the NDR types for a little-endian
 * architecture. This file also depends on the presence of a ANSI
 * C compiler, in that it uses the signed keyword to create the
 * ndr_small_int type.
 */

#ifndef _NDR_TYPES_H
#define  _NDR_TYPES_H

typedef unsigned char 		ndr_boolean;
#define ndr_false       false
#define ndr_true        true
typedef unsigned char 		ndr_byte;

typedef unsigned char 		ndr_char;

typedef signed char 		ndr_small_int;

typedef unsigned char 		ndr_usmall_int;

typedef short int 		ndr_short_int;

typedef unsigned short int	ndr_ushort_int;

typedef long int 		ndr_long_int;

typedef unsigned long int 	ndr_ulong_int;

/*
 * the reps for hyper must match the little-endian NDR rep since
 *  defined(vax) || defined(M_I86) => defined(ALIGNED_SCALAR_ARRAYS)
 */

struct ndr_hyper_int_rep_s_t {
    ndr_ulong_int low;
    ndr_long_int high;
};

struct ndr_uhyper_int_rep_s_t  {
    ndr_ulong_int low;
    ndr_ulong_int high;
};

#ifdef __GNUC__
typedef long long int		ndr_hyper_int;
typedef unsigned long long int	ndr_uhyper_int;
#else
typedef struct ndr_hyper_int_rep_s_t ndr_hyper_int;
typedef struct ndr_uhyper_int_rep_s_t ndr_uhyper_int;
#endif /* __GNUC__ */

typedef float 		        ndr_short_float;
typedef double 			ndr_long_float;

#endif /* _NDR_TYPES_H */
