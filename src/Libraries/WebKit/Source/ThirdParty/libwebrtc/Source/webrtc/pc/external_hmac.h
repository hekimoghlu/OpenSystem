/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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
#ifndef PC_EXTERNAL_HMAC_H_
#define PC_EXTERNAL_HMAC_H_

// External libsrtp HMAC auth module which implements methods defined in
// auth_type_t.
// The default auth module will be replaced only when the ENABLE_EXTERNAL_AUTH
// flag is enabled. This allows us to access to authentication keys,
// as the default auth implementation doesn't provide access and avoids
// hashing each packet twice.

// How will libsrtp select this module?
// Libsrtp defines authentication function types identified by an unsigned
// integer, e.g. SRTP_HMAC_SHA1 is 3. Using authentication ids, the
// application can plug any desired authentication modules into libsrtp.
// libsrtp also provides a mechanism to select different auth functions for
// individual streams. This can be done by setting the right value in
// the auth_type of srtp_policy_t. The application must first register auth
// functions and the corresponding authentication id using
// crypto_kernel_replace_auth_type function.

#include <stdint.h>

#include "third_party/libsrtp/crypto/include/auth.h"
#include "third_party/libsrtp/crypto/include/crypto_types.h"
#include "third_party/libsrtp/include/srtp.h"

#define EXTERNAL_HMAC_SHA1 SRTP_HMAC_SHA1 + 1
#define HMAC_KEY_LENGTH 20

// The HMAC context structure used to store authentication keys.
// The pointer to the key  will be allocated in the external_hmac_init function.
// This pointer is owned by srtp_t in a template context.
typedef struct {
  uint8_t key[HMAC_KEY_LENGTH];
  int key_length;
} ExternalHmacContext;

srtp_err_status_t external_hmac_alloc(srtp_auth_t** a,
                                      int key_len,
                                      int out_len);

srtp_err_status_t external_hmac_dealloc(srtp_auth_t* a);

srtp_err_status_t external_hmac_init(void* state,
                                     const uint8_t* key,
                                     int key_len);

srtp_err_status_t external_hmac_start(void* state);

srtp_err_status_t external_hmac_update(void* state,
                                       const uint8_t* message,
                                       int msg_octets);

srtp_err_status_t external_hmac_compute(void* state,
                                        const uint8_t* message,
                                        int msg_octets,
                                        int tag_len,
                                        uint8_t* result);

srtp_err_status_t external_crypto_init();

#endif  // PC_EXTERNAL_HMAC_H_
