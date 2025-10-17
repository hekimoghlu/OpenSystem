/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 26, 2024.
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

#ifndef _HEIM_ENGINE_H
#define _HEIM_ENGINE_H 1

/* symbol renaming */
#define ENGINE_add_conf_module hc_ENGINE_add_conf_module
#define ENGINE_by_dso hc_ENGINE_by_dso
#define ENGINE_by_id hc_ENGINE_by_id
#define ENGINE_finish hc_ENGINE_finish
#define ENGINE_get_DH hc_ENGINE_get_DH
#define ENGINE_get_RSA hc_ENGINE_get_RSA
#define ENGINE_get_ECDSA hc_ENGINE_get_ECDSA
#define ENGINE_get_RAND hc_ENGINE_get_RAND
#define ENGINE_get_id hc_ENGINE_get_id
#define ENGINE_get_name hc_ENGINE_get_name
#define ENGINE_load_builtin_engines hc_ENGINE_load_builtin_engines
#define ENGINE_set_DH hc_ENGINE_set_DH
#define ENGINE_set_RSA hc_ENGINE_set_RSA
#define ENGINE_set_id hc_ENGINE_set_id
#define ENGINE_set_name hc_ENGINE_set_name
#define ENGINE_set_destroy_function hc_ENGINE_set_destroy_function
#define ENGINE_new hc_ENGINE_new
#define ENGINE_free hc_ENGINE_free
#define ENGINE_up_ref hc_ENGINE_up_ref
#define ENGINE_get_default_DH hc_ENGINE_get_default_DH
#define ENGINE_get_default_RSA hc_ENGINE_get_default_RSA
#define ENGINE_set_default_DH hc_ENGINE_set_default_DH
#define ENGINE_set_default_RSA hc_ENGINE_set_default_RSA
#define ENGINE_set_default_ECDSA hc_ENGINE_set_default_ECDSA
#define ENGINE_get_default_ECDSA hc_ENGINE_get_default_ECDSA

/*
 *
 */

typedef struct hc_engine ENGINE;

#define NID_md2			0
#define NID_md4			1
#define NID_md5			2
#define NID_sha1		4
#define NID_sha256		5
#define NID_sha384		6
#define NID_sha512		7
#define NID_X9_62_prime256v1   415
#define NID_secp160r1   709
#define NID_secp160r2   710
#define OPENSSL_EC_NAMED_CURVE NID_X9_62_prime256v1

/*
 *
 */

#include <hcrypto/rsa.h>
#include <hcrypto/ecdsa.h>
#include <hcrypto/dsa.h>
#include <hcrypto/dh.h>
#include <hcrypto/rand.h>

#define OPENSSL_DYNAMIC_VERSION		(unsigned long)0x00020000

typedef int (*openssl_bind_engine)(ENGINE *, const char *, const void *);
typedef unsigned long (*openssl_v_check)(unsigned long);

ENGINE	*
	ENGINE_new(void);
int ENGINE_free(ENGINE *);
void	ENGINE_add_conf_module(void);
void	ENGINE_load_builtin_engines(void);
ENGINE *ENGINE_by_id(const char *);
ENGINE *ENGINE_by_dso(const char *, const char *);
int	ENGINE_finish(ENGINE *);
int	ENGINE_up_ref(ENGINE *);
int	ENGINE_set_id(ENGINE *, const char *);
int	ENGINE_set_name(ENGINE *, const char *);
int	ENGINE_set_RSA(ENGINE *, const RSA_METHOD *);
int	ENGINE_set_ECDSA(ENGINE *, const ECDSA_METHOD *);
int	ENGINE_set_DH(ENGINE *, const DH_METHOD *);
int	ENGINE_set_destroy_function(ENGINE *, void (*)(ENGINE *));

const char *		ENGINE_get_id(const ENGINE *);
const char *		ENGINE_get_name(const ENGINE *);
const RSA_METHOD *	ENGINE_get_RSA(const ENGINE *);
const ECDSA_METHOD *	ENGINE_get_ECDSA(const ENGINE *);
const DH_METHOD *	ENGINE_get_DH(const ENGINE *);
const RAND_METHOD *	ENGINE_get_RAND(const ENGINE *);

int		ENGINE_set_default_RSA(ENGINE *);
ENGINE *	ENGINE_get_default_RSA(void);
int		ENGINE_set_default_DH(ENGINE *);
ENGINE *	ENGINE_get_default_DH(void);
int		ENGINE_set_default_ECDSA(ENGINE *);
ENGINE *	ENGINE_get_default_ECDSA(void);


#endif /* _HEIM_ENGINE_H */
