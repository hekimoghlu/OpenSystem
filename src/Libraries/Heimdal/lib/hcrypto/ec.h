/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 5, 2025.
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
#ifndef HEIM_EC_H
#define HEIM_EC_H 1

#define EC_GROUP_get_degree hc_EC_GROUP_get_degree
#define EC_KEY_get0_group hc_EC_KEY_get0_group
#define EC_GROUP_get_order hc_EC_GROUP_get_order
#define o2i_ECPublicKey hc_o2i_ECPublicKey
#define EC_KEY_free hc_EC_KEY_free
#define EC_GROUP_new_by_curve_name hc_EC_GROUP_new_by_curve_name
#define EC_KEY_set_group hc_EC_KEY_set_group
#define EC_GROUP_free hc_EC_GROUP_free
#define EC_KEY_check_key hc_EC_KEY_check_key
#define EC_KEY_get0_private_key hc_EC_KEY_get0_private_key
#define EC_KEY_set_private_key hc_EC_KEY_set_private_key

#include <hcrypto/bn.h>
#include <hcrypto/engine.h>

typedef struct EC_KEY EC_KEY;
typedef struct EC_GROUP EC_GROUP;
typedef struct EC_GROUP_ID_s *EC_GROUP_ID;

unsigned long
EC_GROUP_get_degree(EC_GROUP *);

EC_GROUP *
EC_KEY_get0_group(EC_KEY *);

int
EC_GROUP_get_order(EC_GROUP *, BIGNUM *, BN_CTX *);

EC_KEY *
o2i_ECPublicKey(EC_KEY **key, unsigned char **, size_t);

EC_KEY *
EC_KEY_new_by_curve_name(EC_GROUP_ID);

int
EC_KEY_generate_key(EC_KEY *);

void
EC_KEY_free(EC_KEY *);

EC_GROUP *
EC_GROUP_new_by_curve_name(int nid);

void
EC_KEY_set_group(EC_KEY *, EC_GROUP *);

void
EC_GROUP_free(EC_GROUP *);

int
EC_KEY_check_key(const EC_KEY *);

const BIGNUM *EC_KEY_get0_private_key(const EC_KEY *);

int EC_KEY_set_private_key(EC_KEY *, const BIGNUM *);

#endif /* HEIM_EC_H */
