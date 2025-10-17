/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 3, 2022.
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
	File:		comp.h

	Contains:	Glue between core prng code to the Zlib library.

	Written by:	Counterpane, Inc. 

	Copyright: (c) 2000 by Apple Computer, Inc., all rights reserved.

	Change History (most recent first):

		02/10/99	dpm		Created, based on Counterpane source.
 
*/
/* comp.h

   Header for the compression routines added to the Counterpane PRNG. 
*/

#ifndef __YARROW_COMP_H__
#define __YARROW_COMP_H__

#include "smf.h"

/*
 * Kernel version does NULL compression....
 */
#define YARROW_KERNEL

#ifdef	YARROW_KERNEL
/* 
 * Shrink this down to almost nothing to simplify kernel port;
 * with additional hacking on prng.c, this could go away entirely
 */
typedef char COMP_CTX;

/* and define some type3s normally picked up from zlib */
typedef unsigned char Bytef;
typedef unsigned uInt;

#else

#include "zlib.h"

/* Top level compression context */
typedef struct{
	MMPTR buf;
	uInt spaceused;
} COMP_CTX;
#endif	/* YARROW_KERNEL */

typedef enum comp_error_status {
	COMP_SUCCESS = 0,
	COMP_ERR_NULL_POINTER,
	COMP_ERR_LOW_MEMORY,
	COMP_ERR_LIB
} comp_error_status;

/* Exported functions from compress.c */
comp_error_status comp_init(COMP_CTX* ctx);
comp_error_status comp_add_data(COMP_CTX* ctx,Bytef* inp,uInt inplen);
comp_error_status comp_end(COMP_CTX* ctx);
comp_error_status comp_get_ratio(COMP_CTX* ctx,float* out);

#endif
