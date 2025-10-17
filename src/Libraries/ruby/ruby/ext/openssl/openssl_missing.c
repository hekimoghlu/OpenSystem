/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 24, 2023.
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
 * This program is licensed under the same licence as Ruby.
 * (See the file 'LICENCE'.)
 */
#include RUBY_EXTCONF_H

#include <string.h> /* memcpy() */
#if !defined(OPENSSL_NO_ENGINE)
# include <openssl/engine.h>
#endif
#if !defined(OPENSSL_NO_HMAC)
# include <openssl/hmac.h>
#endif
#include <openssl/x509_vfy.h>

#include "openssl_missing.h"

/* added in 1.0.2 */
#if !defined(OPENSSL_NO_EC)
#if !defined(HAVE_EC_CURVE_NIST2NID)
static struct {
    const char *name;
    int nid;
} nist_curves[] = {
    {"B-163", NID_sect163r2},
    {"B-233", NID_sect233r1},
    {"B-283", NID_sect283r1},
    {"B-409", NID_sect409r1},
    {"B-571", NID_sect571r1},
    {"K-163", NID_sect163k1},
    {"K-233", NID_sect233k1},
    {"K-283", NID_sect283k1},
    {"K-409", NID_sect409k1},
    {"K-571", NID_sect571k1},
    {"P-192", NID_X9_62_prime192v1},
    {"P-224", NID_secp224r1},
    {"P-256", NID_X9_62_prime256v1},
    {"P-384", NID_secp384r1},
    {"P-521", NID_secp521r1}
};

int
ossl_EC_curve_nist2nid(const char *name)
{
    size_t i;
    for (i = 0; i < (sizeof(nist_curves) / sizeof(nist_curves[0])); i++) {
	if (!strcmp(nist_curves[i].name, name))
	    return nist_curves[i].nid;
    }
    return NID_undef;
}
#endif
#endif

/*** added in 1.1.0 ***/
#if !defined(HAVE_HMAC_CTX_NEW)
HMAC_CTX *
ossl_HMAC_CTX_new(void)
{
    HMAC_CTX *ctx = OPENSSL_malloc(sizeof(HMAC_CTX));
    if (!ctx)
	return NULL;
    HMAC_CTX_init(ctx);
    return ctx;
}
#endif

#if !defined(HAVE_HMAC_CTX_FREE)
void
ossl_HMAC_CTX_free(HMAC_CTX *ctx)
{
    if (ctx) {
	HMAC_CTX_cleanup(ctx);
	OPENSSL_free(ctx);
    }
}
#endif

#if !defined(HAVE_X509_CRL_GET0_SIGNATURE)
void
ossl_X509_CRL_get0_signature(const X509_CRL *crl, const ASN1_BIT_STRING **psig,
			     const X509_ALGOR **palg)
{
    if (psig != NULL)
	*psig = crl->signature;
    if (palg != NULL)
	*palg = crl->sig_alg;
}
#endif

#if !defined(HAVE_X509_REQ_GET0_SIGNATURE)
void
ossl_X509_REQ_get0_signature(const X509_REQ *req, const ASN1_BIT_STRING **psig,
			     const X509_ALGOR **palg)
{
    if (psig != NULL)
	*psig = req->signature;
    if (palg != NULL)
	*palg = req->sig_alg;
}
#endif
