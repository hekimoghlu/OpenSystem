/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 9, 2025.
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
#ifndef SRTP_CIHPER_PRIV_H
#define SRTP_CIHPER_PRIV_H

#include "cipher.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A trivial platform independent random source.
 * For use in test only.
 */
void srtp_cipher_rand_for_tests(void *dest, uint32_t len);

/*
 * A trivial platform independent 32 bit random number.
 * For use in test only.
 */
uint32_t srtp_cipher_rand_u32_for_tests(void);

#ifdef __cplusplus
}
#endif

#endif /* SRTP_CIPHER_PRIV_H */
