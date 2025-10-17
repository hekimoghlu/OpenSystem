/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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
/* Functions which are used by both single and triple DES enctypes */

#include "krb5_locl.h"

/*
 * A = A xor B. A & B are 8 bytes.
 */

void
_krb5_xor (unsigned char *key, const unsigned char *b)
{
    unsigned char *a = (unsigned char*)key;
    a[0] ^= b[0];
    a[1] ^= b[1];
    a[2] ^= b[2];
    a[3] ^= b[3];
    a[4] ^= b[4];
    a[5] ^= b[5];
    a[6] ^= b[6];
    a[7] ^= b[7];
}

#if defined(DES3_OLD_ENCTYPE) || defined(HEIM_WEAK_CRYPTO)
krb5_error_code
_krb5_des_checksum(krb5_context context,
		   CCDigestAlg alg,
		   struct _krb5_key_data *key,
		   const void *data,
		   size_t len,
		   Checksum *cksum)
{
    struct _krb5_evp_schedule *ctx = key->schedule->data;
    CCDigestRef m;
    unsigned char ivec[8];
    unsigned char *p = cksum->checksum.data;

    krb5_generate_random_block(p, 8);

    m = CCDigestCreate(alg);
    if (m == NULL) {
	krb5_set_error_message(context, ENOMEM, N_("malloc: out of memory", ""));
	return ENOMEM;
    }

    CCDigestUpdate(m, p, 8);
    CCDigestUpdate(m, data, len);
    CCDigestFinal(m, p + 8);
    CCDigestDestroy(m);
    memset (&ivec, 0, sizeof(ivec));
    EVP_CipherInit_ex(&ctx->ectx, NULL, NULL, NULL, (void *)ivec, -1);
    EVP_Cipher(&ctx->ectx, p, p, 24);

    return 0;
}

krb5_error_code
_krb5_des_verify(krb5_context context,
		 CCDigestAlg alg,
		 struct _krb5_key_data *key,
		 const void *data,
		 size_t len,
		 Checksum *C)
{
    struct _krb5_evp_schedule *ctx = key->schedule->data;
    CCDigestRef m;
    unsigned char tmp[24];
    unsigned char res[16];
    unsigned char ivec[8];
    krb5_error_code ret = 0;

    m = CCDigestCreate(alg);
    if (m == NULL) {
	krb5_set_error_message(context, ENOMEM, N_("malloc: out of memory", ""));
	return ENOMEM;
    }

    memset(ivec, 0, sizeof(ivec));
    EVP_CipherInit_ex(&ctx->dctx, NULL, NULL, NULL, (void *)ivec, -1);
    EVP_Cipher(&ctx->dctx, tmp, C->checksum.data, 24);

    CCDigestUpdate(m, tmp, 8); /* confounder */
    CCDigestUpdate(m, data, len);
    CCDigestFinal(m, res);
    CCDigestDestroy(m);
    if(ct_memcmp(res, tmp + 8, sizeof(res)) != 0) {
	krb5_clear_error_message (context);
	ret = KRB5KRB_AP_ERR_BAD_INTEGRITY;
    }
    memset(tmp, 0, sizeof(tmp));
    memset(res, 0, sizeof(res));
    return ret;
}

#endif

static krb5_error_code
RSA_MD5_checksum(krb5_context context,
		 struct _krb5_key_data *key,
		 const void *data,
		 size_t len,
		 unsigned usage,
		 Checksum *C)
{
    if (CCDigest(kCCDigestMD5, data, len, C->checksum.data) != 0)
	krb5_abortx(context, "md5 checksum failed");
    return 0;
}

struct _krb5_checksum_type _krb5_checksum_rsa_md5 = {
    CKSUMTYPE_RSA_MD5,
    "rsa-md5",
    64,
    16,
    F_CPROOF,
    RSA_MD5_checksum,
    NULL
};
