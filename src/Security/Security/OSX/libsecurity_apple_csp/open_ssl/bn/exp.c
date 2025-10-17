/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 22, 2024.
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
/* unused */

#include <stdio.h>
#include <openssl/tmdiff.h>
#include "bn_lcl.h"

#define SIZE	256
#define NUM	(8*8*8)
#define MOD	(8*8*8*8*8)

main(argc,argv)
int argc;
char *argv[];
	{
	BN_CTX ctx;
	BIGNUM a,b,c,r,rr,t,l;
	int j,i,size=SIZE,num=NUM,mod=MOD;
	char *start,*end;
	BN_MONT_CTX mont;
	double d,md;

	BN_MONT_CTX_init(&mont);
	BN_CTX_init(&ctx);
	BN_init(&a);
	BN_init(&b);
	BN_init(&c);
	BN_init(&r);

	start=ms_time_new();
	end=ms_time_new();
	while (size <= 1024*8)
		{
		BN_rand(&a,size,0,0);
		BN_rand(&b,size,1,0);
		BN_rand(&c,size,0,1);

		BN_mod(&a,&a,&c,&ctx);

		ms_time_get(start);
		for (i=0; i<10; i++)
			BN_MONT_CTX_set(&mont,&c,&ctx);
		ms_time_get(end);
		md=ms_time_diff(start,end);

		ms_time_get(start);
		for (i=0; i<num; i++)
			{
			/* bn_mull(&r,&a,&b,&ctx); */
			/* BN_sqr(&r,&a,&ctx); */
			BN_mod_exp_mont(&r,&a,&b,&c,&ctx,&mont);
			}
		ms_time_get(end);
		d=ms_time_diff(start,end)/* *50/33 */;
		printf("%5d bit:%6.2f %6d %6.4f %4d m_set(%5.4f)\n",size,
			d,num,d/num,(int)((d/num)*mod),md/10.0);
		num/=8;
		mod/=8;
		if (num <= 0) num=1;
		size*=2;
		}

	}
