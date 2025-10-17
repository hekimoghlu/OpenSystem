/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
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

#ifdef HEIM_HC_SF

#include <stdio.h>
#include <stdlib.h>
#include <dh.h>

#include <roken.h>

#include <rfc2459_asn1.h>

#include <Security/SecDH.h>

#include "common.h"

struct dh_sf {
    SecDHContext secdh;
};

static int
sf_dh_generate_key(DH *dh)
{
    struct dh_sf *sf = DH_get_ex_data(dh, 0);
    size_t length, size = 0;
    DHParameter dp;
    void *data = NULL;
    int ret;

    if (dh->p == NULL || dh->g == NULL)
	return 0;

    memset(&dp, 0, sizeof(dp));

    ret = _hc_BN_to_integer(dh->p, &dp.prime);
    if (ret == 0)
	ret = _hc_BN_to_integer(dh->g, &dp.base);
    if (ret) {
	free_DHParameter(&dp);
	return 0;
    }
    dp.privateValueLength = NULL;

    ASN1_MALLOC_ENCODE(DHParameter, data, length, &dp, &size, ret);
    free_DHParameter(&dp);
    if (ret)
	return 0;
    if (size != length)
	abort();
    
    if (sf->secdh) {
	SecDHDestroy(sf->secdh);
	sf->secdh = NULL;
    }

    ret = SecDHCreateFromParameters(data, length, &sf->secdh);
    if (ret)
	goto error;

    free(data);
    data = NULL;

    length = BN_num_bytes(dh->p);
    data = malloc(size);
    if (data == NULL)
	goto error;

    ret = SecDHGenerateKeypair(sf->secdh, data, &length);
    if (ret)
	goto error;

    dh->pub_key = BN_bin2bn(data, length, NULL);
    if (dh->pub_key == NULL)
	goto error;

    free(data);

    return 1;

 error:
    if (data)
	free(data);
    if (sf->secdh)
	SecDHDestroy(sf->secdh);
    sf->secdh = NULL;

    return 0;
}

static int
sf_dh_compute_key(unsigned char *shared, const BIGNUM * pub, DH *dh)
{
    struct dh_sf *sf = DH_get_ex_data(dh, 0);
    size_t length, shared_length;
    OSStatus ret;
    void *data;

    shared_length = BN_num_bytes(dh->p);

    length = BN_num_bytes(pub);
    data = malloc(length);
    if (data == NULL)
	return 0;

    BN_bn2bin(pub, data);

    ret = SecDHComputeKey(sf->secdh, data, length, shared, &shared_length);
    free(data);
    if (ret)
	return 0;

    return shared_length;
}


static int
sf_dh_generate_params(DH *dh, int a, int b, BN_GENCB *callback)
{
    return 0;
}

static int
sf_dh_init(DH *dh)
{
    struct dh_sf *sf;

    sf = calloc(1, sizeof(*sf));
    if (sf == NULL)
	return 0;

    DH_set_ex_data(dh, 0, sf);

    return 1;
}

static int
sf_dh_finish(DH *dh)
{
    struct dh_sf *sf = DH_get_ex_data(dh, 0);

    if (sf->secdh)
	SecDHDestroy(sf->secdh);
    free(sf);

    return 1;
}


/*
 *
 */

const DH_METHOD _hc_dh_sf_method = {
    "hcrypto sf DH",
    sf_dh_generate_key,
    sf_dh_compute_key,
    NULL,
    sf_dh_init,
    sf_dh_finish,
    0,
    NULL,
    sf_dh_generate_params
};

/**
 * DH implementation using SecurityFramework
 *
 * @return the DH_METHOD for the DH implementation using SecurityFramework.
 *
 * @ingroup hcrypto_dh
 */

const DH_METHOD *
DH_sf_method(void)
{
    return &_hc_dh_sf_method;
}
#endif /* HEIM_HC_SF */
