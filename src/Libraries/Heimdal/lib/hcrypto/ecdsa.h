/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 26, 2024.
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
#ifndef HEIM_ECDSA_H
#define HEIM_ECDSA_H 1

#define ECDSA_verify hc_ECDSA_verify
#define ECDSA_sign hc_ECDSA_sign
#define ECDSA_size hc_ECDSA_size
#define ECDSA_new hc_ECDSA_new
#define ECDSA_new_method hc_ECDSA_new_method
#define ECDSA_free hc_ECDSA_free
#define ECDSA_public_encrypt hc_ECDSA_public_encrypt
#define ECDSA_public_decrypt hc_ECDSA_public_decrypt
#define ECDSA_private_encrypt hc_ECDSA_private_encrypt
#define ECDSA_private_decrypt hc_ECDSA_private_decrypt
#define ECDSA_set_method hc_ECDSA_set_method
#define ECDSA_get_method hc_ECDSA_get_method

typedef struct ECDSA ECDSA;
typedef struct ECDSA_METHOD ECDSA_METHOD;

#include <hcrypto/ec.h>
#define ECDSA_KEY_SIZE 72

int ECDSA_verify(int, const unsigned char *, unsigned int,
		 unsigned char *, unsigned int, ECDSA *);

int ECDSA_sign(int, const unsigned char *, unsigned int,
	       unsigned char *, unsigned int *, ECDSA *);

int ECDSA_size(EC_KEY *);

int     ECDSA_set_app_data(ECDSA *, void *arg);
void *  ECDSA_get_app_data(const ECDSA *);

struct ECDSA_METHOD {
    const char *name;
    int (*ecdsa_pub_enc)(int,const unsigned char *, unsigned char *, ECDSA *,int);
    int (*ecdsa_pub_dec)(int,const unsigned char *, unsigned char *, ECDSA *,int);
    int (*ecdsa_priv_enc)(int,const unsigned char *, unsigned char *, ECDSA *,int);
    int (*ecdsa_priv_dec)(int,const unsigned char *, unsigned char *, ECDSA *,int);
    int (*init)(ECDSA *ecdsa);
    int (*finish)(ECDSA *ecdsa);
    int flags;
    int (*ecdsa_sign)(int, const unsigned char *, unsigned int,
                    unsigned char *, unsigned int *, const ECDSA *);
    int (*ecdsa_verify)(int, const unsigned char *, unsigned int,
                      unsigned char *, unsigned int, const ECDSA *);
};

struct ECDSA {
    int pad;
    long version;
    const ECDSA_METHOD *meth;
    void *engine;
    struct ecdsa_CRYPTO_EX_DATA {
        void *sk;
        int dummy;
    } ex_data;
    int references;
};

ECDSA *   ECDSA_new(void);
ECDSA *   ECDSA_new_method(ENGINE *);
void    ECDSA_free(ECDSA *);
int     ECDSA_up_ref(ECDSA *);

void    ECDSA_set_default_method(const ECDSA_METHOD *);
const ECDSA_METHOD * ECDSA_get_default_method(void);

const ECDSA_METHOD * ECDSA_get_method(const ECDSA *);
int ECDSA_set_method(ECDSA *, const ECDSA_METHOD *);


#endif /* HEIM_ECDSA_H */
