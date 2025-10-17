/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 16, 2023.
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
#ifndef _CORECRYPTO_CCMODE_GCM_INTERNAL_H_
#define _CORECRYPTO_CCMODE_GCM_INTERNAL_H_

#include <corecrypto/ccmode_impl.h>

#define CCGCM_IV_NBYTES 12
#define CCGCM_BLOCK_NBYTES 16

/* Create a gcm key from a gcm mode object.
 *  key must point to at least sizeof(CCMODE_GCM_KEY(ecb)) bytes of free
 *  storage. */
int ccmode_gcm_init(const struct ccmode_gcm *gcm, ccgcm_ctx *ctx,
    size_t rawkey_len, const void *rawkey);
int ccmode_gcm_set_iv(ccgcm_ctx *ctx, size_t iv_nbytes, const void *iv);
int ccmode_gcm_aad(ccgcm_ctx *ctx, size_t nbytes, const void *in);
int ccmode_gcm_decrypt(ccgcm_ctx *ctx, size_t nbytes, const void *in,
    void *out);
int ccmode_gcm_encrypt(ccgcm_ctx *ctx, size_t nbytes, const void *in,
    void *out);

/*!
 *  @function  ccmode_gcm_finalize() finalizes AES-GCM call sequence
 *  @param key encryption or decryption key
 *  @param tag_nbytes length of tag in bytes
 *  @param tag authentication tag
 *  @result	0=success or non zero= error
 *  @discussion For decryption, the tag parameter must be the expected-tag. A secure compare is performed between the provided expected-tag and the computed-tag. If they are the same, 0 is returned. Otherwise, non zero is returned. For encryption, tag is output and provides the authentication tag.
 *
 */
int ccmode_gcm_finalize(ccgcm_ctx *key, size_t tag_nbytes, void *tag);
int ccmode_gcm_reset(ccgcm_ctx *key);

#define GCM_ECB_KEY_SIZE(ECB_ENCRYPT) \
	((5 * ccn_sizeof_size((ECB_ENCRYPT)->block_size)) \
    + ccn_sizeof_size((ECB_ENCRYPT)->size))

#define CCGCM_FLAGS_INIT_WITH_IV 1

// Here is what the structure looks like in memory
// [ temp space | length | *ecb | *ecb_key | table | ecb_key ]
// size of table depends on the implementation (VNG vs factory)
// currently, VNG and factory share the same "header" described here
// VNG may add additional data after the header
struct _ccmode_gcm_key {
	// 5 blocks of temp space.
	unsigned char H[16];   /* multiplier */
	unsigned char X[16];   /* accumulator */
	unsigned char Y[16];   /* counter */
	unsigned char Y_0[16]; /* initial counter */
	unsigned char buf[16];  /* buffer for stuff */

	// State and length
	uint16_t state;    /* state the GCM code is in */
	uint16_t flags;    /* flags (persistent across reset) */
	uint32_t buf_nbytes; /* length of data in buf */

	uint64_t aad_nbytes; /* 64-bit counter used for IV and AAD */
	uint64_t text_nbytes; /* 64-bit counter for the plaintext PT */

	// ECB
	const struct ccmode_ecb *ecb;          // ecb mode
	// Pointer to the ECB key in the buffer
	void *ecb_key;                         // address of the ecb_key in u, set in init function
	int encdec; //is it an encrypt or decrypt object

	// Buffer with ECB key and H table if applicable
	CC_ALIGNED(16) unsigned char u[]; // ecb key + tables
};

/* Macros for accessing a CCMODE_GCM_KEY.
 *  Common to the generic (factory) and the VNG implementation
 */

#define _CCMODE_GCM_KEY(K) ((struct _ccmode_gcm_key *)(K))
#define CCMODE_GCM_KEY_H(K) (_CCMODE_GCM_KEY(K)->H)
#define CCMODE_GCM_KEY_X(K) (_CCMODE_GCM_KEY(K)->X)
#define CCMODE_GCM_KEY_Y(K) (_CCMODE_GCM_KEY(K)->Y)
#define CCMODE_GCM_KEY_Y_0(K) (_CCMODE_GCM_KEY(K)->Y_0)
#define CCMODE_GCM_KEY_PAD_LEN(K) (_CCMODE_GCM_KEY(K)->buf_nbytes)
#define CCMODE_GCM_KEY_PAD(K) (_CCMODE_GCM_KEY(K)->buf)

#define _CCMODE_GCM_ECB_MODE(K) ((struct _ccmode_gcm_key *)(K))
#define CCMODE_GCM_KEY_ECB(K) (_CCMODE_GCM_ECB_MODE(K)->ecb)
#define CCMODE_GCM_KEY_ECB_KEY(K) ((ccecb_ctx *)_CCMODE_GCM_ECB_MODE(K)->ecb_key)  // set in init function

#define CCMODE_GCM_STATE_IV    1
#define CCMODE_GCM_STATE_AAD   2
#define CCMODE_GCM_STATE_TEXT  3
#define CCMODE_GCM_STATE_FINAL 4

void ccmode_gcm_gf_mult(const unsigned char *a, const unsigned char *b, unsigned char *c);
void ccmode_gcm_gf_mult_32(const unsigned char *a, const unsigned char *b, unsigned char *c);
void ccmode_gcm_gf_mult_64(const unsigned char *a, const unsigned char *b, unsigned char *c);
void ccmode_gcm_mult_h(ccgcm_ctx *key, unsigned char *I);

CC_NONNULL_ALL
void inc_uint(uint8_t *buf, size_t nbytes);

CC_NONNULL_ALL
void ccmode_gcm_update_pad(ccgcm_ctx *key);

CC_NONNULL_ALL
void ccmode_gcm_aad_finalize(ccgcm_ctx *key);

#endif // _CORECRYPTO_CCMODE_GCM_INTERNAL_H_
