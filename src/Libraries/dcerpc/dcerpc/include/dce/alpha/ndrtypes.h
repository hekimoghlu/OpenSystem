/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 2, 2022.
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
/****
 **** NDR types for the Digital ALPHA processor - Little Endian Mode
 ****/

#ifndef _NDRTYPES_H
#define _NDRTYPES_H

typedef unsigned char 		ndr_boolean;

#define ndr_false false
#define ndr_true  true

typedef unsigned char 		ndr_byte;

typedef unsigned char 		ndr_char;

typedef signed char 		ndr_small_int;

typedef unsigned char 		ndr_usmall_int;

typedef short int 		ndr_short_int;

typedef unsigned short int 	ndr_ushort_int;

typedef int 		        ndr_long_int;

typedef unsigned int 	        ndr_ulong_int;

struct ndr_hyper_int_rep_s_t   {
    ndr_long_int high;
    ndr_ulong_int low;
};

struct ndr_uhyper_int_rep_s_t  {
    ndr_ulong_int high;
    ndr_ulong_int low;
};

typedef long int		ndr_hyper_int;
typedef unsigned long int	ndr_uhyper_int;

typedef float			ndr_short_float;
typedef double			ndr_long_float;

#endif /* NDRTYPES_H */
