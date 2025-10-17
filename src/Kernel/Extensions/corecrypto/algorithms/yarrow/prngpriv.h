/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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
	File:		prngpriv.h

	Contains:	Private typedefs and #defines for Counterpane Yarrow PRNG.

	Written by:	Counterpane, Inc. 

	Copyright: (c) 2000 by Apple Computer, Inc., all rights reserved.

	Change History (most recent first):

		02/10/99	dpm		Created, based on Counterpane source.
 
*/
/*
	prngpriv.h

	Completely private header for the Counterpane PRNG. Should only be included by prng.c
*/

#ifndef __YARROW_PRNG_PRIV_H__
#define __YARROW_PRNG_PRIV_H__

#include "userdefines.h"
#include "yarrow.h"
#include "entropysources.h"
#include "comp.h"
#include "sha1mod.h"
#include "smf.h"

#define TOTAL_SOURCES ENTROPY_SOURCES+USER_SOURCES

#ifdef COMPRESSION_ON
#define COMP_SOURCES TOTAL_SOURCES
#else
#define COMP_SOURCES ENTROPY_SOURCES
#endif

/* Error numbers */
typedef enum prng_ready_status {
	PRNG_READY = 33,	/* Compiler will initialize to either 0 or random if allowed to */
	PRNG_NOT_READY = 0
} prng_ready_status;

/* Top level output state */
typedef struct{
	BYTE IV[20];
	BYTE out[20];
	UINT index;			/* current byte to output */
	UINT numout;		/* bytes since last prng_make_new_state */ 
} GEN_CTX;

/* PRNG state structure */
struct PRNG {
	/* Output State */
	GEN_CTX outstate;

	/* Entropy Pools (somewhat unlike a gene pool) */
	YSHA1_CTX pool;
	UINT poolSize[TOTAL_SOURCES];			/* Note that size is in bytes and est in bits */
	UINT poolEstBits[TOTAL_SOURCES];
	COMP_CTX comp_state[COMP_SOURCES];

	/* Status Flags */
	prng_ready_status ready;
};

/*
 * Clients see an opaque PrngRef; internal code uses the 
 * following typedef.
 */
typedef struct PRNG PRNG;


/* Test Macros */
#define CHECKSTATE(p) \
if(p==NULL) {return PRNG_ERR_NOT_READY;} /* Does the state exist? */	\
if(p->ready != PRNG_READY) {return PRNG_ERR_NOT_READY;}	/* Set error state and return */
/* To make sure that a pointer isn't NULL */
#define PCHECK(ptr)  if(ptr==NULL) {return PRNG_ERR_NULL_POINTER;}
/* To make sure that malloc returned a valid value */
#define MCHECK(ptr)  if(ptr==NULL) {return PRNG_ERR_LOW_MEMORY;}
/* To make sure that a given value is non-negative */
#if		defined(macintosh) || defined(__APPLE__)
/* original looks like a bogon */
#define ZCHECK(val)  if(val<0) {return PRNG_ERR_OUT_OF_BOUNDS;}
#else
#define ZCHECK(val)  if(p<0) {return PRNG_ERR_OUT_OF_BOUNDS;}
#endif	/* macintosh */
/* To make sure that the generator state is valid */
#define GENCHECK(p) if(p->outstate.index>20) {return PRNG_ERR_OUT_OF_BOUNDS;} /* index is unsigned */
/* To make sure that the entropy pool is valid */
#define POOLCHECK(p) /* */


#endif
