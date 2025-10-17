/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 15, 2023.
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
#ifndef __content_filter_crypto_h
#define __content_filter_crypto_h

#include <net/content_filter.h>

extern cfil_crypto_state_t
cfil_crypto_init_client(cfil_crypto_key client_key);

extern void
cfil_crypto_cleanup_state(cfil_crypto_state_t state);

extern int
cfil_crypto_sign_data(cfil_crypto_state_t state, cfil_crypto_data_t data,
    const struct iovec *__counted_by(extra_data_count)extra_data, size_t extra_data_count,
    cfil_crypto_signature signature, u_int32_t *signature_length);

#endif // __content_filter_crypto_h
