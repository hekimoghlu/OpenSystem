/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 8, 2024.
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
#include <openssl/bn_legacy.h>
#include <openssl/rand.h>

static int rand(n)
{
    unsigned char x[2];
    RAND_pseudo_bytes(x,2);
    return (x[0] + 2*x[1]);
}

static void bug(char *m, BIGNUM *a, BIGNUM *b)
{
    printf("%s!\na=",m);
    BN_print_fp(stdout, a);
    printf("\nb=");
    BN_print_fp(stdout, b);
    printf("\n");
    fflush(stdout);
}

main()
{
    BIGNUM *a=BN_new(), *b=BN_new(), *c=BN_new(), *d=BN_new(),
	*C=BN_new(), *D=BN_new();
    BN_RECP_CTX *recp=BN_RECP_CTX_new();
    BN_CTX *ctx=BN_CTX_new();

    for(;;) {
	BN_pseudo_rand(a,rand(),0,0);
	BN_pseudo_rand(b,rand(),0,0);
	if (BN_is_zero(b)) continue;

	BN_RECP_CTX_set(recp,b,ctx);
	if (BN_div(C,D,a,b,ctx) != 1)
	    bug("BN_div failed",a,b);
	if (BN_div_recp(c,d,a,recp,ctx) != 1)
	    bug("BN_div_recp failed",a,b);
	else if (BN_cmp(c,C) != 0 || BN_cmp(c,C) != 0)
	    bug("mismatch",a,b);
    }
}
