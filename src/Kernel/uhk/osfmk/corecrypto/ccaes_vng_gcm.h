/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 23, 2022.
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
#ifndef _CORECRYPTO_CCAES_VNG_GCM_H_
#define _CORECRYPTO_CCAES_VNG_GCM_H_

#include <corecrypto/ccaes.h>

#if (CCAES_INTEL_ASM && defined(__x86_64__)) ||                             \
        (CCAES_ARM_ASM && defined(__ARM_NEON__))
#define CCMODE_GCM_VNG_SPEEDUP 1
#else
#define CCMODE_GCM_VNG_SPEEDUP 0
#endif

#include "ccmode_internal.h"

#if CCMODE_GCM_VNG_SPEEDUP

struct _cc_vng_gcm_tables {
#if !defined(__arm64__) && defined(__ARM_NEON__)
	unsigned char Htable[8 * 2] CC_ALIGNED(16);
#else
	unsigned char Htable[16 * 8 * 2] CC_ALIGNED(16);
#endif
};

#define VNG_GCM_TABLE_SIZE sizeof(struct _cc_vng_gcm_tables)
#define CCMODE_GCM_VNG_KEY_Htable(K) (((struct _cc_vng_gcm_tables*)&_CCMODE_GCM_KEY(K)->u[0])->Htable)

int ccaes_vng_gcm_decrypt(ccgcm_ctx *key, size_t nbytes,
    const void *in, void *out);

int ccaes_vng_gcm_encrypt(ccgcm_ctx *key, size_t nbytes,
    const void *in, void *out);

extern void gcm_init(void *Htable, void *H) __asm__("_gcm_init");
extern void gcm_gmult(const void *X, const void *Htable, void *out) __asm__("_gcm_gmult");
extern void gcm_ghash(void *X, void *Htable, const void *in, size_t len) __asm__("_gcm_ghash");
#ifdef  __x86_64__
extern void gcmEncrypt_SupplementalSSE3(const void*, void*, void*, unsigned int, void*, void*) __asm__("_gcmEncrypt_SupplementalSSE3");
extern void gcmDecrypt_SupplementalSSE3(const  void*, void*, void*, unsigned int, void*, void*) __asm__("_gcmDecrypt_SupplementalSSE3");
extern void gcmEncrypt_avx1(const void*, void*, void*, unsigned int, void*, void*) __asm__("_gcmEncrypt_avx1");
extern void gcmDecrypt_avx1(const void*, void*, void*, unsigned int, void*, void*) __asm__("_gcmDecrypt_avx1");
#else
extern void gcmEncrypt(const void*, void*, void*, unsigned int, void*, void*) __asm__("_gcmEncrypt");
extern void gcmDecrypt(const void*, void*, void*, unsigned int, void*, void*) __asm__("_gcmDecrypt");
#endif

/* Use this to statically initialize a ccmode_gcm object for encryption. */
#define CCAES_VNG_GCM_ENCRYPT(ECB_ENCRYPT) { \
.size = ccn_sizeof_size(sizeof(struct _ccmode_gcm_key))  \
+ GCM_ECB_KEY_SIZE(ECB_ENCRYPT)  \
+ VNG_GCM_TABLE_SIZE, \
.block_size = 1, \
.init = ccmode_gcm_init, \
.set_iv = ccmode_gcm_set_iv, \
.gmac = ccmode_gcm_aad, \
.gcm = ccaes_vng_gcm_encrypt, \
.finalize = ccmode_gcm_finalize, \
.reset = ccmode_gcm_reset, \
.custom = (ECB_ENCRYPT), \
.encdec = CCMODE_GCM_ENCRYPTOR\
}

/* Use these function to runtime initialize a ccmode_gcm encrypt object (for
 *  example if it's part of a larger structure). For GCM you always pass a
 *  ecb encrypt mode implementation of some underlying algorithm as the ecb
 *  parameter. */
CC_INLINE
void
ccaes_vng_factory_gcm_encrypt(struct ccmode_gcm *gcm)
{
	struct ccmode_gcm gcm_encrypt = CCAES_VNG_GCM_ENCRYPT(ccaes_ecb_encrypt_mode());
	*gcm = gcm_encrypt;
}

/* Use this to statically initialize a ccmode_gcm object for decryption. */
#define CCAES_VNG_GCM_DECRYPT(ECB_ENCRYPT) { \
.size = ccn_sizeof_size(sizeof(struct _ccmode_gcm_key))  \
+ GCM_ECB_KEY_SIZE(ECB_ENCRYPT)  \
+ VNG_GCM_TABLE_SIZE, \
.block_size = 1, \
.init = ccmode_gcm_init, \
.set_iv = ccmode_gcm_set_iv, \
.gmac = ccmode_gcm_aad, \
.gcm = ccaes_vng_gcm_decrypt, \
.finalize = ccmode_gcm_finalize, \
.reset = ccmode_gcm_reset, \
.custom = (ECB_ENCRYPT), \
.encdec = CCMODE_GCM_DECRYPTOR\
}

/* Use these function to runtime initialize a ccmode_gcm decrypt object (for
 *  example if it's part of a larger structure). For GCM you always pass a
 *  ecb encrypt mode implementation of some underlying algorithm as the ecb
 *  parameter. */
CC_INLINE
void
ccaes_vng_factory_gcm_decrypt(struct ccmode_gcm *gcm)
{
	struct ccmode_gcm gcm_decrypt = CCAES_VNG_GCM_DECRYPT(ccaes_ecb_encrypt_mode());
	*gcm = gcm_decrypt;
}
#endif /* CCMODE_GCM_VNG_SPEEDUP */

#endif /* _CORECRYPTO_CCAES_VNG_GCM_H_ */
