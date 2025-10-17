/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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
#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <dsa.h>

#include <roken.h>

/*
 *
 */

DSA *
DSA_new(void)
{
    DSA *dsa = calloc(1, sizeof(*dsa));
    dsa->meth = rk_UNCONST(DSA_get_default_method());
    dsa->references = 1;
    return dsa;
}

void
DSA_free(DSA *dsa)
{
    if (dsa->references <= 0)
	abort();

    if (--dsa->references > 0)
	return;

    (*dsa->meth->finish)(dsa);

#define free_if(f) if (f) { BN_free(f); }
    free_if(dsa->p);
    free_if(dsa->q);
    free_if(dsa->g);
    free_if(dsa->pub_key);
    free_if(dsa->priv_key);
    free_if(dsa->kinv);
    free_if(dsa->r);
#undef free_if

    memset(dsa, 0, sizeof(*dsa));
    free(dsa);

}

int
DSA_up_ref(DSA *dsa)
{
    return ++dsa->references;
}

/*
 *
 */

static const DSA_METHOD dsa_null_method = {
    "hcrypto null DSA"
};

const DSA_METHOD *
DSA_null_method(void)
{
    return &dsa_null_method;
}


const DSA_METHOD *dsa_default_mech = &dsa_null_method;

void
DSA_set_default_method(const DSA_METHOD *mech)
{
    dsa_default_mech = mech;
}

const DSA_METHOD *
DSA_get_default_method(void)
{
    return dsa_default_mech;
}

int
DSA_verify(int type, const unsigned char * digest, int digest_len,
	   const unsigned char *sig, int sig_len, DSA *dsa)
{
    return -1;
}
