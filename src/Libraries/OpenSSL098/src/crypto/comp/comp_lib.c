/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 23, 2022.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/objects.h>
#include <openssl/comp.h>

COMP_CTX *COMP_CTX_new(COMP_METHOD *meth)
	{
	COMP_CTX *ret;

	if ((ret=(COMP_CTX *)OPENSSL_malloc(sizeof(COMP_CTX))) == NULL)
		{
		/* ZZZZZZZZZZZZZZZZ */
		return(NULL);
		}
	memset(ret,0,sizeof(COMP_CTX));
	ret->meth=meth;
	if ((ret->meth->init != NULL) && !ret->meth->init(ret))
		{
		OPENSSL_free(ret);
		ret=NULL;
		}
	return(ret);
	}

void COMP_CTX_free(COMP_CTX *ctx)
	{
	if(ctx == NULL)
	    return;

	if (ctx->meth->finish != NULL)
		ctx->meth->finish(ctx);

	OPENSSL_free(ctx);
	}

int COMP_compress_block(COMP_CTX *ctx, unsigned char *out, int olen,
	     unsigned char *in, int ilen)
	{
	int ret;
	if (ctx->meth->compress == NULL)
		{
		/* ZZZZZZZZZZZZZZZZZ */
		return(-1);
		}
	ret=ctx->meth->compress(ctx,out,olen,in,ilen);
	if (ret > 0)
		{
		ctx->compress_in+=ilen;
		ctx->compress_out+=ret;
		}
	return(ret);
	}

int COMP_expand_block(COMP_CTX *ctx, unsigned char *out, int olen,
	     unsigned char *in, int ilen)
	{
	int ret;

	if (ctx->meth->expand == NULL)
		{
		/* ZZZZZZZZZZZZZZZZZ */
		return(-1);
		}
	ret=ctx->meth->expand(ctx,out,olen,in,ilen);
	if (ret > 0)
		{
		ctx->expand_in+=ilen;
		ctx->expand_out+=ret;
		}
	return(ret);
	}
