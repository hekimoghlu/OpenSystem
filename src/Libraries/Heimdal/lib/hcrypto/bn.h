/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 23, 2025.
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
 * $Id$
 */

#ifndef _HEIM_BN_H
#define _HEIM_BN_H 1

/* symbol renaming */
#define BN_GENCB_call hc_BN_GENCB_call
#define BN_GENCB_set hc_BN_GENCB_set
#define BN_bin2bn hc_BN_bin2bn
#define BN_bn2bin hc_BN_bn2bin
#define BN_bn2hex hc_BN_bn2hex
#define BN_clear hc_BN_clear
#define BN_clear_bit hc_BN_clear_bit
#define BN_clear_free hc_BN_clear_free
#define BN_cmp hc_BN_cmp
#define BN_dup hc_BN_dup
#define BN_free hc_BN_free
#define BN_is_negative hc_BN_is_negative
#define BN_get_word hc_BN_get_word
#define BN_hex2bn hc_BN_hex2bn
#define BN_is_bit_set hc_BN_is_bit_set
#define BN_new hc_BN_new
#define BN_num_bits hc_BN_num_bits
#define BN_num_bytes hc_BN_num_bytes
#define BN_rand hc_BN_rand
#define BN_set_bit hc_BN_set_bit
#define BN_set_negative hc_BN_set_negative
#define BN_set_word hc_BN_set_word
#define BN_uadd hc_BN_uadd
#define BN_CTX_new hc_BN_CTX_new
#define BN_CTX_free hc_BN_CTX_free
#define BN_CTX_get hc_BN_CTX_get
#define BN_CTX_start hc_BN_CTX_start
#define BN_CTX_end hc_BN_CTX_end

/*
 *
 */

typedef struct BIGNUM BIGNUM;
typedef struct BN_GENCB BN_GENCB;
typedef struct BN_CTX BN_CTX;
typedef struct BN_MONT_CTX BN_MONT_CTX;
typedef struct BN_BLINDING BN_BLINDING;

struct BN_GENCB {
    unsigned int ver;
    void *arg;
    union {
	int (*cb_2)(int, int, BN_GENCB *);
    } cb;
};

/*
 *
 */

BIGNUM *BN_new(void);
void	BN_free(BIGNUM *);
void	BN_clear_free(BIGNUM *);
void	BN_clear(BIGNUM *);
BIGNUM *BN_dup(const BIGNUM *);

int	BN_num_bits(const BIGNUM *);
int	BN_num_bytes(const BIGNUM *);

int	BN_cmp(const BIGNUM *, const BIGNUM *);

void	BN_set_negative(BIGNUM *, int);
int	BN_is_negative(const BIGNUM *);

int	BN_is_bit_set(const BIGNUM *, int);
int	BN_set_bit(BIGNUM *, int);
int	BN_clear_bit(BIGNUM *, int);

int	BN_set_word(BIGNUM *, unsigned long);
unsigned long BN_get_word(const BIGNUM *);

BIGNUM *BN_bin2bn(const void *,int len,BIGNUM *);
int	BN_bn2bin(const BIGNUM *, void *);
int 	BN_hex2bn(BIGNUM **, const char *);
char *	BN_bn2hex(const BIGNUM *);

int	BN_uadd(BIGNUM *, const BIGNUM *, const BIGNUM *);

int	BN_rand(BIGNUM *, int, int, int);

void	BN_GENCB_set(BN_GENCB *, int (*)(int, int, BN_GENCB *), void *);
int	BN_GENCB_call(BN_GENCB *, int, int);

BN_CTX *BN_CTX_new(void);
void	BN_CTX_free(BN_CTX *);
BIGNUM *BN_CTX_get(BN_CTX *);
void	BN_CTX_start(BN_CTX *);
void	BN_CTX_end(BN_CTX *);

#endif
