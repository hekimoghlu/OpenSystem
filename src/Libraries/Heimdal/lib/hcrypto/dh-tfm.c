/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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
#include <dh.h>

#include <roken.h>

#ifdef USE_HCRYPTO_TFM

#include "tfm.h"

static void
BN2mpz(fp_int *s, const BIGNUM *bn)
{
    size_t len;
    void *p;

    len = BN_num_bytes(bn);
    p = malloc(len);
    BN_bn2bin(bn, p);
    fp_read_unsigned_bin(s, p, len);
    free(p);
}


static BIGNUM *
mpz2BN(fp_int *s)
{
    size_t size;
    BIGNUM *bn;
    void *p;

    size = fp_unsigned_bin_size(s);
    p = malloc(size);
    if (p == NULL && size != 0)
	return NULL;
    fp_to_unsigned_bin(s, p);

    bn = BN_bin2bn(p, size, NULL);
    free(p);
    return bn;
}

/*
 *
 */

#define DH_NUM_TRIES 10

static int
tfm_dh_generate_key(DH *dh)
{
    fp_int pub, priv_key, g, p;
    int have_private_key = (dh->priv_key != NULL);
    int codes, times = 0;
    int res;

    if (dh->p == NULL || dh->g == NULL)
	return 0;

    while (times++ < DH_NUM_TRIES) {
	if (!have_private_key) {
	    size_t bits = BN_num_bits(dh->p);

	    if (dh->priv_key)
		BN_free(dh->priv_key);

	    dh->priv_key = BN_new();
	    if (dh->priv_key == NULL)
		return 0;
	    if (!BN_rand(dh->priv_key, bits - 1, 0, 0)) {
		BN_clear_free(dh->priv_key);
		dh->priv_key = NULL;
		return 0;
	    }
	}
	if (dh->pub_key)
	    BN_free(dh->pub_key);

	fp_init_multi(&pub, &priv_key, &g, &p, NULL);

	BN2mpz(&priv_key, dh->priv_key);
	BN2mpz(&g, dh->g);
	BN2mpz(&p, dh->p);

	res = fp_exptmod(&g, &priv_key, &p, &pub);

	fp_zero(&priv_key);
	fp_zero(&g);
	fp_zero(&p);
	if (res != 0)
	    continue;

	dh->pub_key = mpz2BN(&pub);
	fp_zero(&pub);
	if (dh->pub_key == NULL)
	    return 0;

	if (DH_check_pubkey(dh, dh->pub_key, &codes) && codes == 0)
	    break;
	if (have_private_key)
	    return 0;
    }

    if (times >= DH_NUM_TRIES) {
	if (!have_private_key && dh->priv_key) {
	    BN_free(dh->priv_key);
	    dh->priv_key = NULL;
	}
	if (dh->pub_key) {
	    BN_free(dh->pub_key);
	    dh->pub_key = NULL;
	}
	return 0;
    }

    return 1;
}

static int
tfm_dh_compute_key(unsigned char *shared, const BIGNUM * pub, DH *dh)
{
    fp_int s, priv_key, p, peer_pub;
    size_t size = 0;
    int ret;

    if (dh->pub_key == NULL || dh->g == NULL || dh->priv_key == NULL)
	return -1;

    fp_init(&p);
    BN2mpz(&p, dh->p);

    fp_init(&peer_pub);
    BN2mpz(&peer_pub, pub);

    /* check if peers pubkey is reasonable */
    if (fp_isneg(&peer_pub)
	|| fp_cmp(&peer_pub, &p) >= 0
	|| fp_cmp_d(&peer_pub, 1) <= 0)
    {
	fp_zero(&p);
	fp_zero(&peer_pub);
	return -1;
    }

    fp_init(&priv_key);
    BN2mpz(&priv_key, dh->priv_key);

    fp_init(&s);

    ret = fp_exptmod(&peer_pub, &priv_key, &p, &s);

    fp_zero(&p);
    fp_zero(&peer_pub);
    fp_zero(&priv_key);

    if (ret != 0)
	return -1;

    size = fp_unsigned_bin_size(&s);
    fp_to_unsigned_bin(&s, shared);
    fp_zero(&s);

    return size;
}

static int
tfm_dh_generate_params(DH *dh, int a, int b, BN_GENCB *callback)
{
    /* groups should already be known, we don't care about this */
    return 0;
}

static int
tfm_dh_init(DH *dh)
{
    return 1;
}

static int
tfm_dh_finish(DH *dh)
{
    return 1;
}


/*
 *
 */

const DH_METHOD _hc_dh_tfm_method = {
    "hcrypto tfm DH",
    tfm_dh_generate_key,
    tfm_dh_compute_key,
    NULL,
    tfm_dh_init,
    tfm_dh_finish,
    0,
    NULL,
    tfm_dh_generate_params
};

/**
 * DH implementation using tfm.
 *
 * @return the DH_METHOD for the DH implementation using tfm.
 *
 * @ingroup hcrypto_dh
 */

const DH_METHOD *
DH_tfm_method(void)
{
    return &_hc_dh_tfm_method;
}

#endif
