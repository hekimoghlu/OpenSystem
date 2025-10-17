/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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
#include "ec.h"

struct EC_POINT {
    int inf;
    mp_int x;
    mp_int y;
    mp_int z;
};

struct EC_GROUP {
    size_t size;
    mp_int prime;
    mp_int order;
    mp_int Gx;
    mp_int Gy;
};

struct EC_KEY {
    int type;
    EC_GROUP *group;
    EC_POINT *pubkey;
    mp_int privkey;
};


unsigned long
EC_GROUP_get_degree(EC_GROUP *)
{
}

EC_GROUP *
EC_KEY_get0_group(EC_KEY *)
{
}

int
EC_GROUP_get_order(EC_GROUP *, BIGNUM *, BN_CTX *)
{
}

EC_KEY *
o2i_ECPublicKey(EC_KEY **key, unsigned char **, size_t)
{
}

void
EC_KEY_free(EC_KEY *)
{

}

EC_GROUP *
EC_GROUP_new_by_curve_name(int nid)
{
}

EC_KEY *
EC_KEY_new_by_curve_name(EC_GROUP_ID nid)
{
    EC_KEY *key;

    key = calloc(1, sizeof(*key));
    return key;
}

void
EC_POINT_free(EC_POINT *p)
{
    mp_clear_multi(&p->x, p->y, p->z, NULL);
    free(p);
}

static int
ec_point_mul(EC_POINT *res, const EC_GROUP *group, const mp_int *point)
{
}

EC_POINT *
EC_POINT_new(void)
{
    EC_POINT *p;

    p = calloc(1, sizeof(*p));

    if (mp_init_multi(&p->x, &p->y, &p->z, NULL) != 0) {
	EC_POINT_free(p);
	return NULL;
    }

    return p;
}

int
EC_KEY_generate_key(EC_KEY *key)
{
    int ret = 0;

    if (key->group == NULL)
	return 0;

    do {
	random(key->privkey, key->group->size);
    } while(mp_cmp(key->privkey, key->group->order) >= 0);

    if (key->pubkey == NULL)
	key->pubkey = EC_POINT_new();

    if (ec_point_mul(&key->pubkey, key->group, key->privkey) != 1)
	goto error;

    ret = 1;
 error:
    ECPOINT_free(&base);

    return ret;
}

void
EC_KEY_set_group(EC_KEY *, EC_GROUP *)
{

}

void
EC_GROUP_free(EC_GROUP *)
{
}

int
EC_KEY_check_key(const EC_KEY *)
{
}

const BIGNUM *
EC_KEY_get0_private_key(const EC_KEY *key)
{
}

int
EC_KEY_set_private_key(EC_KEY *key, const BIGNUM *bn)
{
}
