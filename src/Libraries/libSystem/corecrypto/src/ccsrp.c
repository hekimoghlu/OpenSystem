/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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

#include <corecrypto/ccsrp.h>
#include <corecrypto/ccstubs.h>

bool ccsrp_client_verify_session(ccsrp_ctx_t srp, const uint8_t *HAMK_bytes) {
    CC_STUB(false);
}

int ccsrp_generate_verifier(ccsrp_ctx_t srp, const char *username, size_t password_len,
                            const void *password, size_t salt_len, const void *salt,
                            void *verifier) {
    CC_STUB_ERR();
}

int ccsrp_client_start_authentication(ccsrp_ctx_t srp, struct ccrng_state *rng, void *A_bytes) {
    printf("DARLING CRYPTO STUB: %s\n", __PRETTY_FUNCTION__);
}

int ccsrp_client_process_challenge(ccsrp_ctx_t srp, const void *username, size_t password_len,
                                   const void *password, size_t salt_len, const void *salt,
                                   const void *B_bytes, void *M_bytes) {
    CC_STUB_ERR();
}
