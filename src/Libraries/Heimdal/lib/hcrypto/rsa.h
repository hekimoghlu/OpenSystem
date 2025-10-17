/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 18, 2023.
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

#ifndef _HEIM_RSA_H
#define _HEIM_RSA_H 1

/* symbol renaming */
#define RSA_null_method hc_RSA_null_method
#define RSA_imath_method hc_RSA_imath_method
#define RSA_cdsa_method hc_RSA_cdsa_method
#define RSA_sf_method hc_RSA_sf_method
#define RSA_tfm_method hc_RSA_tfm_method
#define RSA_ltm_method hc_RSA_ltm_method
#define RSA_gmp_method hc_RSA_gmp_method
#define RSA_tfm_method hc_RSA_tfm_method
#define RSA_new hc_RSA_new
#define RSA_new_method hc_RSA_new_method
#define RSA_free hc_RSA_free
#define RSA_up_ref hc_RSA_up_ref
#define RSA_set_default_method hc_RSA_set_default_method
#define RSA_get_default_method hc_RSA_get_default_method
#define RSA_set_method hc_RSA_set_method
#define RSA_get_method hc_RSA_get_method
#define RSA_set_app_data hc_RSA_set_app_data
#define RSA_get_app_data hc_RSA_get_app_data
#define RSA_check_key hc_RSA_check_key
#define RSA_size hc_RSA_size
#define RSA_public_encrypt hc_RSA_public_encrypt
#define RSA_public_decrypt hc_RSA_public_decrypt
#define RSA_private_encrypt hc_RSA_private_encrypt
#define RSA_private_decrypt hc_RSA_private_decrypt
#define RSA_sign hc_RSA_sign
#define RSA_verify hc_RSA_verify
#define RSA_generate_key_ex hc_RSA_generate_key_ex
#define d2i_RSAPrivateKey hc_d2i_RSAPrivateKey
#define i2d_RSAPrivateKey hc_i2d_RSAPrivateKey
#define i2d_RSAPublicKey hc_i2d_RSAPublicKey
#define d2i_RSAPublicKey hc_d2i_RSAPublicKey

/*
 *
 */

typedef struct RSA RSA;
typedef struct RSA_METHOD RSA_METHOD;

#include <hcrypto/bn.h>
#include <hcrypto/engine.h>

struct RSA_METHOD {
    const char *name;
    int (*rsa_pub_enc)(int,const unsigned char *, unsigned char *, RSA *,int);
    int (*rsa_pub_dec)(int,const unsigned char *, unsigned char *, RSA *,int);
    int (*rsa_priv_enc)(int,const unsigned char *, unsigned char *, RSA *,int);
    int (*rsa_priv_dec)(int,const unsigned char *, unsigned char *, RSA *,int);
    void *rsa_mod_exp;
    void *bn_mod_exp;
    int (*init)(RSA *rsa);
    int (*finish)(RSA *rsa);
    int flags;
    char *app_data;
    int (*rsa_sign)(int, const unsigned char *, unsigned int,
		    unsigned char *, unsigned int *, const RSA *);
    int (*rsa_verify)(int, const unsigned char *, unsigned int,
		      unsigned char *, unsigned int, const RSA *);
    int (*rsa_keygen)(RSA *, int, BIGNUM *, BN_GENCB *);
};

struct RSA {
    int pad;
    long version;
    const RSA_METHOD *meth;
    void *engine;
    BIGNUM *n;
    BIGNUM *e;
    BIGNUM *d;
    BIGNUM *p;
    BIGNUM *q;
    BIGNUM *dmp1;
    BIGNUM *dmq1;
    BIGNUM *iqmp;
    struct rsa_CRYPTO_EX_DATA {
	void *sk;
	int dummy;
    } ex_data;
    int references;
    int flags;
    void *_method_mod_n;
    void *_method_mod_p;
    void *_method_mod_q;

    char *bignum_data;
    void *blinding;
    void *mt_blinding;
};

#define RSA_FLAG_NO_BLINDING		0x0080

#define RSA_PKCS1_PADDING		1
#define RSA_PKCS1_OAEP_PADDING		4
#define RSA_PKCS1_PADDING_SIZE		11

/*
 *
 */

const RSA_METHOD *RSA_null_method(void);
const RSA_METHOD *RSA_gmp_method(void);
const RSA_METHOD *RSA_sf_method(void);
const RSA_METHOD *RSA_cdsa_method(void);
const RSA_METHOD *RSA_tfm_method(void);
const RSA_METHOD *RSA_ltm_method(void);

/*
 *
 */

RSA *	RSA_new(void);
RSA *	RSA_new_method(ENGINE *);
void	RSA_free(RSA *);
int	RSA_up_ref(RSA *);

void	RSA_set_default_method(const RSA_METHOD *);
const RSA_METHOD * RSA_get_default_method(void);

const RSA_METHOD * RSA_get_method(const RSA *);
int RSA_set_method(RSA *, const RSA_METHOD *);

int	RSA_set_app_data(RSA *, void *arg);
void *	RSA_get_app_data(const RSA *);

int	RSA_check_key(const RSA *);
int	RSA_size(const RSA *);

int	RSA_public_encrypt(int,const unsigned char*,unsigned char*,RSA *,int);
int	RSA_private_encrypt(int,const unsigned char*,unsigned char*,RSA *,int);
int	RSA_public_decrypt(int,const unsigned char*,unsigned char*,RSA *,int);
int	RSA_private_decrypt(int,const unsigned char*,unsigned char*,RSA *,int);

int RSA_sign(int, const unsigned char *, unsigned int,
	     unsigned char *, unsigned int *, RSA *);
int RSA_verify(int, const unsigned char *, unsigned int,
	       unsigned char *, unsigned int, RSA *);

int	RSA_generate_key_ex(RSA *, int, BIGNUM *, BN_GENCB *);

RSA *	d2i_RSAPrivateKey(RSA *, const unsigned char **, size_t);
int	i2d_RSAPrivateKey(RSA *, unsigned char **);

int	i2d_RSAPublicKey(RSA *, unsigned char **);
RSA *	d2i_RSAPublicKey(RSA *, const unsigned char **, size_t);

#endif /* _HEIM_RSA_H */
