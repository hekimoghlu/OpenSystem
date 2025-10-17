/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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
	File:		comp.c

	Contains:	NULL compression. Kernel version of Yarrow assumes
				incoming seed data is truly random.
*/
#include "WindowsTypesForMac.h"
#include "comp.h"

#ifdef	YARROW_KERNEL

/* null compression */
comp_error_status comp_init(__unused COMP_CTX* ctx)
{
	return COMP_SUCCESS;
}


comp_error_status comp_add_data( __unused COMP_CTX* ctx, 
								 __unused Bytef* inp, 
								 __unused uInt inplen )
{
	return COMP_SUCCESS;
}

comp_error_status comp_get_ratio( __unused COMP_CTX* ctx,float* out )
{
	*out = 1.0;
	return COMP_SUCCESS;
}

comp_error_status comp_end( __unused COMP_CTX* ctx )
{
	return COMP_SUCCESS;
}

#else

/* original Yarrow compression, must be linked with zlib */

#if		defined(macintosh) || defined(__APPLE__)
#include "WindowsTypesForMac.h"
#include "yarrowUtils.h"
#include <string.h>
#include <stdlib.h>
#else
#include <windows.h>
#endif
#include <math.h>
#include "comp.h"

/* Check that the pointer is not NULL */
#define PCHECK(ptr)  if(ptr==NULL) {return COMP_ERR_NULL_POINTER;}
#define MMPCHECK(mmptr) if(mmptr==MM_NULL) {return COMP_ERR_NULL_POINTER;}
/* Check that the important parts of the context are ok */
#define CTXCHECK(ctx) \
PCHECK(ctx)				\
MMPCHECK(ctx->buf)

/* Might want to vary these by context */
#define BUFSIZE  16384 /* 16K */
#define OUTBUFSIZE 16800 /* = inbufsize*1.01 + 12 (See zlib docs) */
#define SHIFTSIZE 4096 /* BUFSIZE/4 */

#define _MIN(a,b) (((a)<(b))?(a):(b))


/* Initialize these routines */
comp_error_status comp_init(COMP_CTX* ctx)
{
	ctx->buf = mmMalloc(BUFSIZE);
	if(ctx->buf == MM_NULL) {goto cleanup_comp_init;}
	ctx->spaceused = 0;

	return COMP_SUCCESS;

cleanup_comp_init:
	mmFree(ctx->buf);

	return COMP_ERR_LOW_MEMORY;
}


comp_error_status comp_add_data(COMP_CTX* ctx,Bytef* inp,uInt inplen)
{
	uInt shifts;
	uInt blocksize;
	BYTE* buf;

	CTXCHECK(ctx);
	PCHECK(inp);

	buf = (BYTE*)mmGetPtr(ctx->buf);

	if(inplen+SHIFTSIZE>BUFSIZE)
	{
		blocksize = _MIN(inplen,BUFSIZE);
		memmove(buf,inp,blocksize);
		ctx->spaceused = blocksize;
	}
	else
	{
		if(inplen+ctx->spaceused>BUFSIZE) 
		{
			shifts = (uInt)ceil((inplen+ctx->spaceused-BUFSIZE)/(float)SHIFTSIZE);
			blocksize = _MIN(shifts*SHIFTSIZE,ctx->spaceused);
			memmove(buf,buf+blocksize,BUFSIZE-blocksize);
			ctx->spaceused = ctx->spaceused - blocksize;
		}
		memmove(buf+ctx->spaceused,inp,inplen);
		ctx->spaceused += inplen;
	}

	return COMP_SUCCESS;
}

comp_error_status comp_get_ratio(COMP_CTX* ctx,float* out)
{
	Bytef *inbuf,*outbuf;
	uLong insize,outsize;
	int resp;

	*out = 0;

	CTXCHECK(ctx);
	PCHECK(out);

	if(ctx->spaceused == 0) {return COMP_SUCCESS;}

	inbuf = (Bytef*)mmGetPtr(ctx->buf);
	outbuf = (Bytef*)malloc(OUTBUFSIZE);
	if(outbuf==NULL) {return COMP_ERR_LOW_MEMORY;}

	insize = ctx->spaceused;
	outsize = OUTBUFSIZE;

	resp = compress(outbuf,&outsize,inbuf,insize);
	if(resp==Z_MEM_ERROR) {return COMP_ERR_LOW_MEMORY;}
	if(resp==Z_BUF_ERROR) {return COMP_ERR_LIB;}

	*out = (float)outsize/(float)insize;

	/* Thrash the memory and free it */
	trashMemory(outbuf, OUTBUFSIZE);
	free(outbuf);

	return COMP_SUCCESS;
}

comp_error_status comp_end(COMP_CTX* ctx)
{
	if(ctx == NULL) {return COMP_SUCCESS;} /* Since nothing is left undone */

	mmFree(ctx->buf);
	ctx->buf = MM_NULL;

	return COMP_SUCCESS;
}

#endif	/* YARROW_KERNEL */

