/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 28, 2024.
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
#ifndef CIHPER_TYPES_H
#define CIHPER_TYPES_H

#include "cipher.h"
#include "auth.h"

/*
 * cipher types that can be included in the kernel
 */

extern const srtp_cipher_type_t srtp_null_cipher;
extern const srtp_cipher_type_t srtp_aes_icm_128;
extern const srtp_cipher_type_t srtp_aes_icm_256;
#ifdef GCM
extern const srtp_cipher_type_t srtp_aes_icm_192;
extern const srtp_cipher_type_t srtp_aes_gcm_128;
extern const srtp_cipher_type_t srtp_aes_gcm_256;
#endif

/*
 * auth func types that can be included in the kernel
 */

extern const srtp_auth_type_t srtp_null_auth;
extern const srtp_auth_type_t srtp_hmac;

/*
 * other generic debug modules that can be included in the kernel
 */

extern srtp_debug_module_t srtp_mod_auth;
extern srtp_debug_module_t srtp_mod_cipher;
extern srtp_debug_module_t srtp_mod_stat;
extern srtp_debug_module_t srtp_mod_alloc;

/* debug modules for cipher types */
extern srtp_debug_module_t srtp_mod_aes_icm;
#ifdef OPENSSL
extern srtp_debug_module_t srtp_mod_aes_gcm;
#endif
#ifdef NSS
extern srtp_debug_module_t srtp_mod_aes_gcm;
#endif

/* debug modules for auth types */
extern srtp_debug_module_t srtp_mod_hmac;

#endif
