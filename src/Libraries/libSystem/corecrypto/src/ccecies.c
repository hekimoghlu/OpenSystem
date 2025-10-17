/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 23, 2025.
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

#include <corecrypto/ccecies.h>
#include <corecrypto/ccstubs.h>

size_t ccecies_encrypt_gcm_ciphertext_size(ccec_pub_ctx_t public_key,
		ccecies_gcm_t ecies, size_t plaintext_len) {
	CC_STUB(1);
}

int
ccecies_encrypt_gcm( ccec_pub_ctx_t public_key,
		const ccecies_gcm_t ecies,
		size_t plaintext_len,
		const uint8_t *plaintext,
		size_t sharedinfo1_byte_len,
		const void *sharedinfo_1,
		size_t sharedinfo2_byte_len,
		const void *sharedinfo_2,
		size_t *encrypted_blob_len,
		uint8_t *encrypted_blob) {
    CC_STUB_ERR();
}

int
ccecies_decrypt_gcm(ccec_full_ctx_t full_key,
		const ccecies_gcm_t ecies,
		size_t encrypted_blob_len,
		const uint8_t *encrypted_blob,
		size_t sharedinfo1_byte_len,
		const void *sharedinfo_1,
		size_t sharedinfo2_byte_len,
		const void *sharedinfo_2,
		size_t *plaintext_len,
		uint8_t *plaintext) {
	CC_STUB_ERR();
}

size_t
ccecies_decrypt_gcm_plaintext_size(ccec_full_ctx_t full_key,
		ccecies_gcm_t ecies,
		size_t ciphertext_len) {
	CC_STUB_ERR();
}

void
ccecies_encrypt_gcm_setup(ccecies_gcm_t ecies,
		const struct ccdigest_info *di,
		struct ccrng_state *rng,
		const struct ccmode_gcm *aes_gcm_enc,
		uint32_t cipher_key_size,
		uint32_t mac_tag_size,
		uint32_t options) {
	CC_STUB_VOID()
}
